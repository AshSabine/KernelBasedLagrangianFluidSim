use piston_window::*;
use wgpu;

mod sim;
use sim::{
	FluidSimulation,
    FluidInitialState
};

async fn main() {
	//	Create a Piston window
    let mut window: PistonWindow = WindowSettings::new("Fluid Simulation", [800, 600])
        .exit_on_esc(true)
        .build()
        .unwrap();

    //  Get device stuff
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let surface = unsafe { instance.create_surface(&window) }.unwrap();

    let adapter = instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        },
    ).await.unwrap();


    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: if cfg!(target_arch = "wasm32") {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                wgpu::Limits::default()
            },
            
        },
        None, // Trace path
    ).await.unwrap();

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
    while let Some(e) = window.next() {
        if let Some(args) = e.render_args() {
            // Run the fluid simulation
            fluid_simulation.compute();
            
            // Get the current frame
            //let frame = surface.get_current_texture().unwrap();

            // Render the particles
            fluid_simulation.render_particles(
                &mut encoder, 
                frame
            );
        }
    }
}