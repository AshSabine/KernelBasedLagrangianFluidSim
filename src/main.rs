use winit::{
    event::{
        Event, WindowEvent,
        WindowEvent::{KeyboardInput}, //, KeyboardInput, VirtualKeyCode, 
        ElementState
    },
    window::WindowBuilder,
    event_loop::{
        ControlFlow, 
        EventLoop
    },

};

use pollster;
use wgpu;

mod sim;
use sim::{
	FluidSimulation,
    FluidInitialState
};

fn main() {
	//  Create an event loop + winit window
    let event_loop: EventLoop<()> = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

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
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
            }, None,
        ).await.unwrap()
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Render Encoder"),
    });

	// Initialize the FluidSimulation
    let initial_state = FluidInitialState {
        pos: vec![nalgebra::Vector2::new(100.0, 100.0); sim::MAX_PARTICLES],
        vel: vec![nalgebra::Vector2::new(100.0, 100.0); sim::MAX_PARTICLES],
    };
    
    let mut fluid_simulation = FluidSimulation::new(
        device,
        queue,
        initial_state,
        &mut window,
    );

	// Main loop
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        },
        Event::MainEventsCleared => {
            // Update simulation state
            fluid_simulation.compute();

            // Render the simulation
            let output = surface.get_current_texture().unwrap();
            let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

            fluid_simulation.render_particles(&mut encoder, &view);

            queue.submit(std::iter::once(encoder.finish()));

            // Request a redraw
            window.request_redraw();
        }
        _ => {}
    });

    /* 
    while let Some(e) = window.next() {
        if let Some(args) = e.render_args() {
            // Run the fluid simulation
            fluid_simulation.compute();
            
            // Get the current frame
            let output = surface.get_current_texture().unwrap();
            let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Render the particles
            fluid_simulation.render_particles(
                &mut encoder, 
                &view
            );
        }
    }
    */
}