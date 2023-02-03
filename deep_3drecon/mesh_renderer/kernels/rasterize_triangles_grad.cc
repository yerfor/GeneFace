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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace {

// Threshold for a barycentric coordinate triplet's sum, below which the
// coordinates at a pixel are deemed degenerate. Most such degenerate triplets
// in an image will be exactly zero, as this is how pixels outside the mesh
// are rendered.
constexpr float kDegenerateBarycentricCoordinatesCutoff = 0.9f;

// If the area of a triangle is very small in screen space, the corner vertices
// are approaching colinearity, and we should drop the gradient to avoid
// numerical instability (in particular, blowup, as the forward pass computation
// already only has 8 bits of precision).
constexpr float kMinimumTriangleArea = 1e-13;

}  // namespace

namespace tf_mesh_renderer {

  using ::tensorflow::DEVICE_CPU;
  using ::tensorflow::OpKernel;
  using ::tensorflow::OpKernelConstruction;
  using ::tensorflow::OpKernelContext;
  using ::tensorflow::PartialTensorShape;
  using ::tensorflow::Status;
  using ::tensorflow::Tensor;
  using ::tensorflow::TensorShape;
  using ::tensorflow::errors::InvalidArgument;

  REGISTER_OP("RasterizeTrianglesGrad")
      .Input("vertices: float32")
      .Input("triangles: int32")
      .Input("barycentric_coordinates: float32")
      .Input("triangle_ids: int32")
      .Input("df_dbarycentric_coordinates: float32")
      .Attr("image_width: int")
      .Attr("image_height: int")
      .Output("df_dvertices: float32");

  class RasterizeTrianglesGradOp : public OpKernel {
   public:
    explicit RasterizeTrianglesGradOp(OpKernelConstruction* context)
        : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("image_width", &image_width_));
      OP_REQUIRES(context, image_width_ > 0,
                  InvalidArgument("Image width must be > 0, got ", image_width_));

      OP_REQUIRES_OK(context, context->GetAttr("image_height", &image_height_));
      OP_REQUIRES(
          context, image_height_ > 0,
          InvalidArgument("Image height must be > 0, got ", image_height_));
    }

    ~RasterizeTrianglesGradOp() override {}

    void Compute(OpKernelContext* context) override {
      const Tensor& vertices_tensor = context->input(0);
      OP_REQUIRES(
          context,
          PartialTensorShape({-1, 4}).IsCompatibleWith(vertices_tensor.shape()),
          InvalidArgument(
              "RasterizeTrianglesGrad expects vertices to have shape (-1, 4)."));
      auto vertices_flat = vertices_tensor.flat<float>();
      const unsigned int vertex_count = vertices_flat.size() / 4;
      const float* vertices = vertices_flat.data();

      const Tensor& triangles_tensor = context->input(1);
      OP_REQUIRES(
          context,
          PartialTensorShape({-1, 3}).IsCompatibleWith(triangles_tensor.shape()),
          InvalidArgument(
              "RasterizeTrianglesGrad expects triangles to be a matrix."));
      auto triangles_flat = triangles_tensor.flat<int>();
      const int* triangles = triangles_flat.data();

      const Tensor& barycentric_coordinates_tensor = context->input(2);
      OP_REQUIRES(context,
                  TensorShape({image_height_, image_width_, 3}) ==
                      barycentric_coordinates_tensor.shape(),
                  InvalidArgument(
                      "RasterizeTrianglesGrad expects barycentric_coordinates to "
                      "have shape {image_height, image_width, 3}"));
      auto barycentric_coordinates_flat =
          barycentric_coordinates_tensor.flat<float>();
      const float* barycentric_coordinates = barycentric_coordinates_flat.data();

      const Tensor& triangle_ids_tensor = context->input(3);
      OP_REQUIRES(
          context,
          TensorShape({image_height_, image_width_}) ==
              triangle_ids_tensor.shape(),
          InvalidArgument(
              "RasterizeTrianglesGrad expected triangle_ids to have shape "
              " {image_height, image_width}"));
      auto triangle_ids_flat = triangle_ids_tensor.flat<int>();
      const int* triangle_ids = triangle_ids_flat.data();

      // The naming convention we use for all derivatives is d<y>_d<x> ->
      // the partial of y with respect to x.
      const Tensor& df_dbarycentric_coordinates_tensor = context->input(4);
      OP_REQUIRES(
          context,
          TensorShape({image_height_, image_width_, 3}) ==
              df_dbarycentric_coordinates_tensor.shape(),
          InvalidArgument(
              "RasterizeTrianglesGrad expects df_dbarycentric_coordinates "
              "to have shape {image_height, image_width, 3}"));
      auto df_dbarycentric_coordinates_flat =
          df_dbarycentric_coordinates_tensor.flat<float>();
      const float* df_dbarycentric_coordinates =
          df_dbarycentric_coordinates_flat.data();

      Tensor* df_dvertices_tensor = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, TensorShape({vertex_count, 4}),
                                              &df_dvertices_tensor));
      auto df_dvertices_flat = df_dvertices_tensor->flat<float>();
      float* df_dvertices = df_dvertices_flat.data();
      std::fill(df_dvertices, df_dvertices + vertex_count * 4, 0.0f);

      // We first loop over each pixel in the output image, and compute
      // dbarycentric_coordinate[0,1,2]/dvertex[0x, 0y, 1x, 1y, 2x, 2y].
      // Next we compute each value above's contribution to
      // df/dvertices, building up that matrix as the output of this iteration.
      for (unsigned int pixel_id = 0; pixel_id < image_height_ * image_width_;
           ++pixel_id) {
        // b0, b1, and b2 are the three barycentric coordinate values
        // rendered at pixel pixel_id.
        const float b0 = barycentric_coordinates[3 * pixel_id];
        const float b1 = barycentric_coordinates[3 * pixel_id + 1];
        const float b2 = barycentric_coordinates[3 * pixel_id + 2];

        if (b0 + b1 + b2 < kDegenerateBarycentricCoordinatesCutoff) {
          continue;
        }

        const float df_db0 = df_dbarycentric_coordinates[3 * pixel_id];
        const float df_db1 = df_dbarycentric_coordinates[3 * pixel_id + 1];
        const float df_db2 = df_dbarycentric_coordinates[3 * pixel_id + 2];

        const int triangle_at_current_pixel = triangle_ids[pixel_id];
        const int* vertices_at_current_pixel =
            &triangles[3 * triangle_at_current_pixel];

        // Extract vertex indices for the current triangle.
        const int v0_id = 4 * vertices_at_current_pixel[0];
        const int v1_id = 4 * vertices_at_current_pixel[1];
        const int v2_id = 4 * vertices_at_current_pixel[2];

        // Extract x,y,w components of the vertices' clip space coordinates.
        const float x0 = vertices[v0_id];
        const float y0 = vertices[v0_id + 1];
        const float w0 = vertices[v0_id + 3];
        const float x1 = vertices[v1_id];
        const float y1 = vertices[v1_id + 1];
        const float w1 = vertices[v1_id + 3];
        const float x2 = vertices[v2_id];
        const float y2 = vertices[v2_id + 1];
        const float w2 = vertices[v2_id + 3];

        // Compute pixel's NDC-s.
        const int ix = pixel_id % image_width_;
        const int iy = pixel_id / image_width_;
        const float px = 2 * (ix + 0.5f) / image_width_ - 1.0f;
        const float py = 2 * (iy + 0.5f) / image_height_ - 1.0f;

        // Baricentric gradients wrt each vertex coordinate share a common factor.
        const float db0_dx = py * (w1 - w2) - (y1 - y2);
        const float db1_dx = py * (w2 - w0) - (y2 - y0);
        const float db2_dx = -(db0_dx + db1_dx);
        const float db0_dy = (x1 - x2) - px * (w1 - w2);
        const float db1_dy = (x2 - x0) - px * (w2 - w0);
        const float db2_dy = -(db0_dy + db1_dy);
        const float db0_dw = px * (y1 - y2) - py * (x1 - x2);
        const float db1_dw = px * (y2 - y0) - py * (x2 - x0);
        const float db2_dw = -(db0_dw + db1_dw);

        // Combine them with chain rule.
        const float df_dx = df_db0 * db0_dx + df_db1 * db1_dx + df_db2 * db2_dx;
        const float df_dy = df_db0 * db0_dy + df_db1 * db1_dy + df_db2 * db2_dy;
        const float df_dw = df_db0 * db0_dw + df_db1 * db1_dw + df_db2 * db2_dw;

        // Values of edge equations and inverse w at the current pixel.
        const float edge0_over_w = x2 * db0_dx + y2 * db0_dy + w2 * db0_dw;
        const float edge1_over_w = x2 * db1_dx + y2 * db1_dy + w2 * db1_dw;
        const float edge2_over_w = x1 * db2_dx + y1 * db2_dy + w1 * db2_dw;
        const float w_inv = edge0_over_w + edge1_over_w + edge2_over_w;

        // All gradients share a common denominator.
        const float w_sqr = 1 / (w_inv * w_inv);

        // Gradients wrt each vertex share a common factor.
        const float edge0 = w_sqr * edge0_over_w;
        const float edge1 = w_sqr * edge1_over_w;
        const float edge2 = w_sqr * edge2_over_w;

        df_dvertices[v0_id + 0] += edge0 * df_dx;
        df_dvertices[v0_id + 1] += edge0 * df_dy;
        df_dvertices[v0_id + 3] += edge0 * df_dw;
        df_dvertices[v1_id + 0] += edge1 * df_dx;
        df_dvertices[v1_id + 1] += edge1 * df_dy;
        df_dvertices[v1_id + 3] += edge1 * df_dw;
        df_dvertices[v2_id + 0] += edge2 * df_dx;
        df_dvertices[v2_id + 1] += edge2 * df_dy;
        df_dvertices[v2_id + 3] += edge2 * df_dw;
      }
    }

   private:
    int image_width_;
    int image_height_;
  };

  REGISTER_KERNEL_BUILDER(Name("RasterizeTrianglesGrad").Device(DEVICE_CPU),
                          RasterizeTrianglesGradOp);

}  // namespace tf_mesh_renderer
