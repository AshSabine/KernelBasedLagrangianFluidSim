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

    //  Get device stuff
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let surface = unsafe { instance.create_surface(&window) }.unwrap();

    let (device, queue) = pollster::block_on( async {
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

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

	//  Initialize the FluidSimulation
    let initial_state = FluidInitialState {
        pos: vec![nalgebra::Vector2::new(100.0, 100.0); sim::MAX_PARTICLES],
        vel: vec![nalgebra::Vector2::new(100.0, 100.0); sim::MAX_PARTICLES],
    };
    
    let mut fluid_simulation = FluidSimulation::new(
        &device,
        initial_state,
        &window,
    );

	//  Main loop
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
                    
                    //  Update sim + render
                    fluid_simulation.compute(&mut encoder);
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