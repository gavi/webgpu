const rows = 5000
const cols = 5000

// Define global buffer size
const BUFFER_SIZE = new Float32Array(rows*cols).byteLength; 

// Compute shader
const shader = `
@group(0) @binding(0)
var<storage, read> input1: array<f32>;
@group(0) @binding(1)
var<storage, read> input2: array<f32>;
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id)
  global_id : vec3u,

  @builtin(local_invocation_id)
  local_id : vec3u,
) {
  // Avoid accessing the buffer out of bounds
  if (global_id.x >= ${BUFFER_SIZE}) {
    return;
  }

  output[global_id.x] = input1[global_id.x]+input2[global_id.x];
}
`;

// Main function

async function init() {
  const startTime = performance.now();
  // 1: request adapter and device
  if (!navigator.gpu) {
    throw Error('WebGPU not supported.');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw Error('Couldn\'t request WebGPU adapter.');
  }

  adapter.requestAdapterInfo().then((info)=>{
    console.log(info)
    document.write("<h2>Adapter Info</h2>")
    document.write("<pre>vendor: "+info.vendor+"")
    document.write("<pre>architecture: "+info.architecture+"</pre>")
  })

  const device = await adapter.requestDevice();

  // 2: Create a shader module from the shader template literal
  const shaderModule = device.createShaderModule({
    code: shader
  });

// 3: Create input (input1, input2) and  output (output) buffers to read GPU calculations to, and a staging (stagingBuffer) buffer to be mapped for JavaScript access  
  const matrixA = new Float32Array(rows*cols).map((_,i) => 1);
  const matrixB = new Float32Array(rows*cols).map((_,i) => 2);

  const input1 = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(input1, 0, matrixA, 0, matrixA.length);

  const input2 = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(input2, 0, matrixB, 0, matrixB.length);


  const output = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  const stagingBuffer = device.createBuffer({
    size: BUFFER_SIZE,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });

  // 4: Create a GPUBindGroupLayout to define the bind group structure, create a GPUBindGroup from it,
  // then use it to create a GPUComputePipeline

  const bindGroupLayout =
  device.createBindGroupLayout({
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage"
        }
    },{
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage"
        }
    },{
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "storage"
      }
    }]
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{
        binding: 0,
        resource: {
          buffer: input1,
        }
      },{
        binding: 1,
        resource: {
          buffer: input2,
        }
      },{
      binding: 2,
      resource: {
        buffer: output,
      }
    }]
  });

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    }),
    compute: {
      module: shaderModule,
      entryPoint: 'main'
    }
  });

  // 5: Create GPUCommandEncoder to issue commands to the GPU
  const commandEncoder = device.createCommandEncoder();

  // 6: Initiate render pass
  const passEncoder = commandEncoder.beginComputePass();
    
  // 7: Issue commands
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(1);//65535

  // End the render pass
  passEncoder.end();

  // Copy output buffer to staging buffer
  commandEncoder.copyBufferToBuffer(
    output,
    0, // Source offset
    stagingBuffer,
    0, // Destination offset
    BUFFER_SIZE
  );

  // 8: End frame by passing array of command buffers to command queue for execution
  device.queue.submit([commandEncoder.finish()]);

  // map staging buffer to read results back to JS
  await stagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    BUFFER_SIZE // Length
  );

  const copyArrayBuffer = stagingBuffer.getMappedRange(0, BUFFER_SIZE);
  const data = copyArrayBuffer.slice();
  stagingBuffer.unmap();
  const endTime = performance.now();
  console.log(new Float32Array(data));
  const executionTime = endTime - startTime;
  document.write('Execution time: ', executionTime, 'milliseconds');
  console.log(device)
}

init().catch((e)=>{
    alert(e)
});


