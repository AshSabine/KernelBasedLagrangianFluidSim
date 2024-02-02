use winit::{
	event::*, event_loop::{
		EventLoop
	}, 
	keyboard::{KeyCode, PhysicalKey}, 
	window::{Window, WindowBuilder}
};

use pollster;
use wgpu;

mod sim;
use sim::{
	FluidSimulation,
	FluidInitialState
};

//	Struct
struct State<'a> {
	surface: wgpu::Surface<'a>,
	device: wgpu::Device,
	queue: wgpu::Queue,
	config: wgpu::SurfaceConfiguration,
	size: winit::dpi::PhysicalSize<u32>,
	window: &'a Window,

	fluid_sim: FluidSimulation,
}

impl<'a> State<'a> {
	async fn new(
		window: &'a Window
	) -> State<'a> {
		let size = window.inner_size();

		//  This is the instance, a handle to the GPU. 
		//	Used to create everything else needed.
		let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
			backends: wgpu::Backends::all(),
			..Default::default()
		});

		//	This is the inner part of the window.
		let surface = { instance.create_surface(window) }.unwrap();

		//  This is the actual interface w/ the GPU.
		let adapter = instance.request_adapter(
			&wgpu::RequestAdapterOptions {
				power_preference: wgpu::PowerPreference::default(),
				compatible_surface: Some(&surface),
				force_fallback_adapter: false,
			},
		).await.unwrap();
	
		//  Get the device and queue (used to send/queue operations)
		let (device, queue) = pollster::block_on( async {
			adapter.request_device(
				&wgpu::DeviceDescriptor {
					label: None,
					required_features: wgpu::Features::VERTEX_WRITABLE_STORAGE,
					required_limits: if cfg!(target_arch = "wasm32") {
						wgpu::Limits::downlevel_webgl2_defaults()
					} else {
						wgpu::Limits::default()
					},
				}, None,
			).await.unwrap()
		});
	
		//  Configure surface
		let surface_caps = surface.get_capabilities(&adapter);
		let surface_format = surface_caps.formats.iter()
			.copied()
			.filter(|f| f.is_srgb())
			.next()
			.unwrap_or(surface_caps.formats[0]);
		
		let config = wgpu::SurfaceConfiguration {
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			format: surface_format,
			width: 800, //window.inner_size().width,	//800
			height: 600, //window.inner_size().height,	//600
			present_mode: surface_caps.present_modes[0],
			alpha_mode: surface_caps.alpha_modes[0],
			view_formats: vec![],
			desired_maximum_frame_latency: 100,
		};
		surface.configure(&device, &config);

		//		Fluid sim stuff
		//  Initial state
		let initial_state = FluidInitialState {
			pos: vec![nalgebra::Vector2::new(100.0, 100.0); sim::MAX_PARTICLES],
			vel: vec![nalgebra::Vector2::new(1.0, 1.0); sim::MAX_PARTICLES],
		};

		//  Create
		let fluid_sim = FluidSimulation::new(
			&device,
			initial_state,
		);

		Self {
			surface,
			device,
			queue,
			config,
			size,
			window,
			fluid_sim,
		}
	}

	fn render(
		&mut self
	) -> Result<(), wgpu::SurfaceError> {
		let output = self.surface.get_current_texture()?;
		let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("Render Encoder"),
		});

		//  Update sim + copy data
		self.fluid_sim.compute(&mut encoder);
		self.fluid_sim.copy_to_vertex_buffer(&mut encoder);

		//	Render
		self.fluid_sim.render_particles(&mut encoder, &view);
		
		//  Submit to the queue
		self.queue.submit(std::iter::once(encoder.finish()));

		//	Present output
		output.present();

		//	Return ok
		Ok(())
	}

	pub fn resize(
		&mut self, new_size: winit::dpi::PhysicalSize<u32>
	) {
		if new_size.width > 0 && new_size.height > 0 {
			self.size = new_size;
			self.config.width = new_size.width;
			self.config.height = new_size.height;
			self.surface.configure(&self.device, &self.config);
		}
	}
}


//	Funcs
pub async fn run() {	
	//  Create an event loop + winit window
	let event_loop: EventLoop<()> = EventLoop::new().unwrap();
	let window = WindowBuilder::new().build(&event_loop).unwrap();

	//	New state
	let mut state = State::new(&window).await;

	//	  Main loop
	if let Err(err) = event_loop.run(|event, target| 
		match event {
			Event::WindowEvent {
				ref event,
				window_id,
			} if window_id == window.id() => match event {
				WindowEvent::CloseRequested | 
				WindowEvent::KeyboardInput {
					event: KeyEvent {
						state: ElementState::Pressed,
						physical_key: PhysicalKey::Code(KeyCode::Escape),
						..
					}, ..
				} => target.exit(),
				WindowEvent::RedrawRequested => {
					match state.render() {
						Ok(_) => {}
						Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
							state.resize(state.size)
						}
						Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
						Err(wgpu::SurfaceError::Timeout) => println!("Surface timeout"),
					}
	
					//  Request a redraw
					window.request_redraw();
				},
				WindowEvent::Resized(physical_size) => {
					state.resize(*physical_size);
				},
				_ => {}
			},
			
			_ => {}
		}
	) {
		println!("error: {err:?}");
	}
}
