//	Vertex input
struct VertexInput {
    @location(0) position: vec2<f32>,
};

//	Output from vertex shader to fragment shader
struct	VertexOutput {
    @builtin(position) position: vec4<f32>,
};

//	Fragment shader input
struct FragmentInput {
    @builtin(position) position: vec4<f32>,
};

//	Main vertex shader function
@vertex
fn main_vertex(
	input: VertexInput
) -> VertexOutput {
    var output: VertexOutput;
    // Assuming z and w are 0 for 2D points
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    return output;
}

//	Main fragment shader function
@fragment
fn main_fragment(
	input: FragmentInput
) -> @location(0) vec4<f32> {
    // For simplicity, render particles as white points
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}