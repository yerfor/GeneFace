// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <vector>

#include "rasterize_triangles_impl.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tf_mesh_renderer {

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::int32;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;

REGISTER_OP("RasterizeTriangles")
    .Input("vertices: float32")
    .Input("triangles: int32")
    .Attr("image_width: int")
    .Attr("image_height: int")
    .Output("barycentric_coordinates: float32")
    .Output("triangle_ids: int32")
    .Output("z_buffer: float32")
    .Doc(R"doc(
Implements a rasterization kernel for rendering mesh geometry.

vertices: 2-D tensor with shape [vertex_count, 4]. The 3-D positions of the mesh
  vertices in clip-space (XYZW).
triangles: 2-D tensor with shape [triangle_count, 3]. Each row is a tuple of
  indices into vertices specifying a triangle to be drawn. The triangle has an
  outward facing normal when the given indices appear in a clockwise winding to
  the viewer.
image_width: positive int attribute specifying the width of the output image.
image_height: positive int attribute specifying the height of the output image.
barycentric_coordinates: 3-D tensor with shape [image_height, image_width, 3]
  containing the rendered barycentric coordinate triplet per pixel, before
  perspective correction. The triplet is the zero vector if the pixel is outside
  the mesh boundary. For valid pixels, the ordering of the coordinates
  corresponds to the ordering in triangles.
triangle_ids: 2-D tensor with shape [image_height, image_width]. Contains the
  triangle id value for each pixel in the output image. For pixels within the
  mesh, this is the integer value in the range [0, num_vertices] from triangles.
  For vertices outside the mesh this is 0; 0 can either indicate belonging to
  triangle 0, or being outside the mesh. This ensures all returned triangle ids
  will validly index into the vertex array, enabling the use of tf.gather with
  indices from this tensor. The barycentric coordinates can be used to determine
  pixel validity instead.
z_buffer: 2-D tensor with shape [image_height, image_width]. Contains the Z
  coordinate in Normalized Device Coordinates for each pixel occupied by a
  triangle.
)doc");

class RasterizeTrianglesOp : public OpKernel {
 public:
  explicit RasterizeTrianglesOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("image_width", &image_width_));
    OP_REQUIRES(context, image_width_ > 0,
                InvalidArgument("Image width must be > 0, got ", image_width_));

    OP_REQUIRES_OK(context, context->GetAttr("image_height", &image_height_));
    OP_REQUIRES(
        context, image_height_ > 0,
        InvalidArgument("Image height must be > 0, got ", image_height_));
  }

  ~RasterizeTrianglesOp() override {}

  void Compute(OpKernelContext* context) override {
    const Tensor& vertices_tensor = context->input(0);
    OP_REQUIRES(
        context,
        PartialTensorShape({-1, 4}).IsCompatibleWith(vertices_tensor.shape()),
        InvalidArgument(
            "RasterizeTriangles expects vertices to have shape (-1, 4)."));
    auto vertices_flat = vertices_tensor.flat<float>();
    const float* vertices = vertices_flat.data();

    const Tensor& triangles_tensor = context->input(1);
    OP_REQUIRES(
        context,
        PartialTensorShape({-1, 3}).IsCompatibleWith(triangles_tensor.shape()),
        InvalidArgument(
            "RasterizeTriangles expects triangles to be a matrix."));
    auto triangles_flat = triangles_tensor.flat<int32>();
    const int32* triangles = triangles_flat.data();
    const int triangle_count = triangles_flat.size() / 3;

    Tensor* barycentric_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({image_height_, image_width_, 3}),
                       &barycentric_tensor));

    Tensor* triangle_ids_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({image_height_, image_width_}),
                                &triangle_ids_tensor));

    Tensor* z_buffer_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                2, TensorShape({image_height_, image_width_}),
                                &z_buffer_tensor));

    // Clear barycentric and triangle id buffers to 0.
    // Clear z-buffer to 1 (the farthest NDC z value).
    barycentric_tensor->flat<float>().setZero();
    triangle_ids_tensor->flat<int32>().setZero();
    z_buffer_tensor->flat<float>().setConstant(1);

    RasterizeTrianglesImpl(vertices, triangles, triangle_count, image_width_,
                           image_height_,
                           triangle_ids_tensor->flat<int32>().data(),
                           barycentric_tensor->flat<float>().data(),
                           z_buffer_tensor->flat<float>().data());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RasterizeTrianglesOp);

  int image_width_;
  int image_height_;
};

REGISTER_KERNEL_BUILDER(Name("RasterizeTriangles").Device(DEVICE_CPU),
                        RasterizeTrianglesOp);

}  // namespace tf_mesh_renderer
