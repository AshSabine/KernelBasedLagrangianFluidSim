//use piston_window::PistonWindow;
//		Import
use winit::window::Window;
use wgpu::{
	util::DeviceExt,
	Device, Queue,
};


use nalgebra::{Vector2, Vector3};

//      Data
pub const MAX_PARTICLES: usize = 2000000;

//		Structures
#[repr(C)]
#[derive(Clone, Debug)]
pub struct FluidInitialState {	//	SoA because this allows me to have seperate pos and vel buffers
    pub pos: Vec<Vector2<f32>>,
    pub vel: Vec<Vector2<f32>>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Settings {
    //      Particle stuff
    max_particles: usize, 

	//		Timekeeping	
	time_scale: f32,
	framerate: i32,

	//		Physics
	gravity: f32,

	prediction_factor: f32,

	smoothing_radius: f32,
	target_density: f32,
    
	pressure_multiplier: f32,
    pressure_multiplier_near: f32,

    viscosity_strength: f32,

	//		Bounds & obstacles
	bounds_size: Vector2<f32>,
    obstacle_size: Vector2<f32>,
    obstacle_pos: Vector2<f32>,

	collision_damping: f32,
}

pub struct FluidSimulation {
	num_particles: u32,

	//		Simulation data
	//	Buffers
	position_buffer: wgpu::Buffer,		//	Vector2<f32>
	velocity_buffer: wgpu::Buffer,		//	Vector2<f32>

	predicted_pos_buffer: wgpu::Buffer,	//	Vector2<f32>

	density_buffer:	wgpu::Buffer,		//	f32

	local_indices_buffer: wgpu::Buffer,	//	Vector3<u32>
	local_offsets_buffer: wgpu::Buffer,	//	u32

    //  Pipelines
    pipeline_update_pos: wgpu::ComputePipeline,

    pipeline_external_forces: wgpu::ComputePipeline,
    pipeline_predict_pos: wgpu::ComputePipeline,

    pipeline_update_locality: wgpu::ComputePipeline,
    pipeline_update_density: wgpu::ComputePipeline,

    pipeline_apply_pressure: wgpu::ComputePipeline,
    pipeline_apply_viscosity: wgpu::ComputePipeline,

	//	Bind Groups
	buffers_bind_group: wgpu::BindGroup,
	settings_bind_group: wgpu::BindGroup,

	//	Other
	settings: Settings,

	//		Rendering data
	particle_render_shader: wgpu::ShaderModule,
    particle_render_pipeline: wgpu::RenderPipeline,

	//		Extraneous data
	device: Device,
    queue: Queue,
}

impl FluidSimulation {
	pub fn new(
        device: Device, 
        queue: Queue,
        initial_state: FluidInitialState, 
        window: &mut Window
    ) -> Self {
		//			Simulation
		//		Create buffers
		//	Basic data
        let position_buffer = create_buffer(&device, &initial_state.pos, "Positions Buffer");
        let velocity_buffer = create_buffer(&device, &initial_state.vel, "Velocities Buffer");

        //  Other physics
        let predicted_pos_buffer = create_buffer_zeros::<Vector2<f32>>(&device, MAX_PARTICLES, "Predicted Positions Buffer");
        let density_buffer = create_buffer_zeros::<Vector2<f32>>(&device, MAX_PARTICLES, "Densities Buffer");

        //  Spacial hashing stuff
        let local_indices_buffer = create_buffer_zeros::<Vector3<u32>>(&device, MAX_PARTICLES, "Local Indices Buffer");
        let local_offsets_buffer = create_buffer_zeros::<u32>(&device, MAX_PARTICLES, "Local Offsets Buffer");

        //  Settings
        let settings = Settings {
            max_particles: MAX_PARTICLES, 

			//		Timekeeping	
			time_scale: 1.,
			framerate: 30,

			//		Physics
			gravity: 9.81,

			prediction_factor: 1./120.,

			smoothing_radius: 2.,
			target_density: 60.,
			
			pressure_multiplier: 0.5,
			pressure_multiplier_near: 0.5,

			viscosity_strength: 0.1,

			//		Bounds & obstacles
			bounds_size: Vector2::new(800., 600.),
			obstacle_size: Vector2::new(100., 100.),
			obstacle_pos: Vector2::new(400., 200.),

			collision_damping: 0.5,
        };

        let settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Settings Buffer"),
            contents: bytemuck::cast_slice(&[settings]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        //      Shader modules
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders.wgsl"));

        //      Bind group
        let buffers_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                //  Position
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                //  Velocity
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                //  Predicted position
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                //  Density
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                //  Local indices
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                //  Local offsets
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("Buffers Bind Group Layout"),
        });
        let buffers_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &buffers_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: predicted_pos_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: density_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: local_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: local_offsets_buffer.as_entire_binding(),
                },
            ],
            label: Some("Buffers Bind Group"),
        });

        let settings_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::all(),
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("Settings Bind Group Layout"),
        });
        let settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SimulationSettings Bind Group"),
            layout: &settings_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: settings_buffer.as_entire_binding()
            }],
        });

        //      Pipelines
        let layouts = &[&buffers_layout, &settings_layout];
        
        let pipeline_update_pos = create_compute_helper(
            &device, layouts, &shader, 
            "UpdatePositionsCompute", "Update Pos Pipeline"
        );

        let pipeline_external_forces = create_compute_helper(
            &device, layouts, &shader, 
            "ApplyExternalForcesCompute", "External Forces Pipeline"
        );

        let pipeline_predict_pos = create_compute_helper(
            &device, layouts, &shader, 
            "UpdatePredictedPosCompute", "Predict Pos Pipeline"
        );

        let pipeline_update_locality = create_compute_helper(
            &device, layouts, &shader, 
            "UpdateSpatialHashCompute", "Update Locality Pipeline"
        );

        let pipeline_update_density = create_compute_helper(
            &device, layouts, &shader, 
            "UpdateDensityCompute", "Update Density Pipeline"
        );

        let pipeline_apply_pressure = create_compute_helper(
            &device, layouts, &shader, 
            "ApplyPressureForceCompute", "Apply Pressure Pipeline"
        );

        let pipeline_apply_viscosity = create_compute_helper(
            &device, layouts, &shader, 
            "ApplyViscosityForceCompute", "Apply Viscosity Pipeline"
        );

		//			Rendering
		//	Create shader module
		let render_shader = device.create_shader_module(wgpu::include_wgsl!("render.wgsl"));

		//	Create layout
		let render_layout =
		device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("Render Pipeline Layout"),
			bind_group_layouts: &[],
			push_constant_ranges: &[],
		});

		//	Create render pipeline for particle rendering
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: "main_vertex",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: "main_fragment",
                targets: &[Some( wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE), // Adjust blend state as needed
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
			
			primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,

                unclipped_depth: false,
                conservative: false,
            },
			
			depth_stencil: None, // 1.
			multisample: wgpu::MultisampleState {
				count: 1, // 2.
				mask: !0, // 3.
				alpha_to_coverage_enabled: false, // 4.
			},
			multiview: None, // 5.

			label: None,
        });

		//	Create object
		FluidSimulation {
            num_particles: 0 as u32,
			
			//		Simulation data
            //	Buffers
			position_buffer,
            velocity_buffer,

            predicted_pos_buffer,
            density_buffer,

            local_indices_buffer,
            local_offsets_buffer,

			//	Pipelines
			pipeline_update_pos,

    		pipeline_external_forces,
    		pipeline_predict_pos,

    		pipeline_update_locality,
    		pipeline_update_density,

   			pipeline_apply_pressure,
    		pipeline_apply_viscosity,

			//	Bind Groups
			buffers_bind_group,
			settings_bind_group,

			//	Other
            settings,

			//		Render data
			particle_render_shader: render_shader,
			particle_render_pipeline: render_pipeline,

			//		Other data
            device,
            queue,
        }
    }

	//		Compute stuff
	pub fn compute(&mut self) {
		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { 
			label: Some("Compute Encoder") 
		});
		
		let dispatch_size: u32 = (self.settings.max_particles / 64) as u32;

		//		Compute steps
        //	Update particle positions. This is the first thing we do.
		self.run_compute_pass(encoder, self.pipeline_update_pos, dispatch_size);

		//	Apply external forces to the particles.
		self.run_compute_pass(encoder, self.pipeline_external_forces, dispatch_size);		

		//	Predict where the particles are going to be next frame.
		self.run_compute_pass(encoder, self.pipeline_predict_pos, dispatch_size);

		//	Update the spacial hash.
		self.run_compute_pass(encoder, self.pipeline_update_locality, dispatch_size);

		//	Update all densities.
		self.run_compute_pass(encoder, self.pipeline_update_density, dispatch_size);

		//	Calculate pressure force
		self.run_compute_pass(encoder, self.pipeline_apply_pressure, dispatch_size);

		//	Calculate viscosity force
		self.run_compute_pass(encoder, self.pipeline_apply_viscosity, dispatch_size);

		//		Finishing
    	//	Submit the command encoder
    	self.queue.submit(std::iter::once(encoder.finish()));
	}

	fn run_compute_pass(
		&mut self,
		mut encoder: wgpu::CommandEncoder,
		pipeline: wgpu::ComputePipeline,
		dispatch_size: u32,
	) {
		{
			let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
				label: None, //Some("") 
                timestamp_writes: None,
			});
			cpass.set_pipeline(&pipeline);
			cpass.set_bind_group(0, &self.buffers_bind_group, &[]);
			cpass.set_bind_group(0, &self.settings_bind_group, &[]);
			cpass.dispatch_workgroups(dispatch_size, 1, 1);
		}
	}

	//		Render stuff
	//	This function does a single render pass.
    pub fn render_particles(
        &self, 
        encoder: &mut wgpu::CommandEncoder, 
        view: &wgpu::TextureView
    ) {{
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some( wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // Set pipeline and bind groups
        render_pass.set_pipeline(&self.particle_render_pipeline);
        render_pass.set_bind_group(0, &self.buffers_bind_group, &[]);
        render_pass.set_bind_group(1, &self.settings_bind_group, &[]);

        // Draw particles
        render_pass.draw(0..self.num_particles, 0..1);
	}}
}

//		Utilities
//	Buffer creation function
fn create_buffer<T: bytemuck::Pod>(
	device: &wgpu::Device, 
	data: &[T],
    label: &str
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn create_buffer_zeros<T: bytemuck::Pod + bytemuck::Zeroable + Clone>(
    device: &wgpu::Device,
    count: usize,
    label: &str,
) -> wgpu::Buffer {
    let zeroed_data: Vec<T> = vec![T::zeroed(); count];

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&zeroed_data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

//	Compute pipeline function
fn create_compute_helper(
    device: &wgpu::Device,
	layouts: &[&wgpu::BindGroupLayout],
    module: &wgpu::ShaderModule,
    entry_point: &str,
    label: &str,
) -> wgpu::ComputePipeline {
	device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: layouts,
            push_constant_ranges: &[],
        })),
        module: &module,
        entry_point: entry_point,
    })
}