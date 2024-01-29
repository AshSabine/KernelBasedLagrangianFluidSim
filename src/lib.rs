use winit::{
    event::*, event_loop::{
        ControlFlow, 
        EventLoop
    }, keyboard::{KeyCode, PhysicalKey}, window::WindowBuilder
};

use pollster;
use wgpu;

mod sim;
use sim::{
	FluidSimulation,
    FluidInitialState
};

pub async fn run() {
	//  Create an event loop + winit window
    let event_loop: EventLoop<()> = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    //  This is the GPU instance
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let surface = unsafe { instance.create_surface(&window) }.unwrap();

    //  This is the interface with the GPU
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
        width: 800,
        height: 600,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 100,
    };
    surface.configure(&device, &config);

    //      Fluid sim stuff
	//  Initial state
    let initial_state = FluidInitialState {
        pos: vec![nalgebra::Vector2::new(100.0, 100.0); sim::MAX_PARTICLES],
        vel: vec![nalgebra::Vector2::new(100.0, 100.0); sim::MAX_PARTICLES],
    };
    
    //  Create
    let mut fluid_simulation = FluidSimulation::new(
        &device,
        initial_state,
    );

	//      Main loop
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
                    let output = surface.get_current_texture().unwrap();
                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Render Encoder"),
                    }); 
                    
                    //  Update sim + copy data
                    fluid_simulation.compute(&mut encoder);
					fluid_simulation.copy_to_vertex_buffer(&mut encoder);

					//	Render
                    fluid_simulation.render_particles(&mut encoder, &view);
                    
                    //  Submit to the queue
                    queue.submit(std::iter::once(encoder.finish()));
    
                    //  Request a redraw
                    window.request_redraw();
                }
                _ => {}
            },
            
            _ => {}
        }
    ) {
        println!("error: {err:?}");
    }
}