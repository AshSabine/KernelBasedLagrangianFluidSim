use pollster;

fn main() {
	pollster::block_on(fluid_sim::run())
}