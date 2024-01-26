use piston_window::*;

mod sim;
use sim::{
	FluidSimulation,
};

fn main() {
	//	Create a Piston window
    let mut window: PistonWindow = WindowSettings::new("Fluid Simulation", [800, 600])
        .exit_on_esc(true)
        .build()
        .unwrap();

	// Initialize the FluidSimulation
    let mut fluid_simulation = FluidSimulation::new(window.factory.clone());

	// Main loop
    while let Some(e) = window.next() {
        if let Some(args) = e.render_args() {
            // Run the fluid simulation
            fluid_simulation.run();

            // Render the particles
            fluid_simulation.render(&mut window, args);
        }
    }
}