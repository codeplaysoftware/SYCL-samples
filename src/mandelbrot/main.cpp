/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Description:
 *    Application description for Mandelbrot demo.
 *
 **************************************************************************/


#include "mandel.hpp"

#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/ImageView.h>
#include <Magnum/Shaders/FlatGL.h>
#include <Magnum/Primitives/Square.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Trade/MeshData.h>

#include <sycl/sycl.hpp>

constexpr size_t WIDTH = 800;
constexpr size_t HEIGHT = 600;
constexpr Magnum::PixelFormat PIXELFORMAT{Magnum::PixelFormat::RGBA8Unorm};
const size_t PIXELDATASIZE{WIDTH*HEIGHT*Magnum::pixelFormatSize(PIXELFORMAT)};

class MandelbrotApp : public Magnum::Platform::Application
{
  // Use doubles for more zoom
  MandelbrotCalculator m_calc;

  // Texture for displaying the set
  Magnum::GL::Texture2D m_tex;

  // Mesh and shader to draw the texture
  Magnum::GL::Mesh m_mesh;
  Magnum::Shaders::FlatGL2D m_shader;

  // Coordinates of the center point
  double m_ctr_x = 0;
  double m_ctr_y = 0;

  // The viewable range on Y axis
  double m_range = 1;

  // Mouse coordinates from previous click
  double m_prev_mx = 0;
  double m_prev_my = 0;

 public:
  MandelbrotApp(const Arguments& arguments)
  : Magnum::Platform::Application{
      arguments,
      Configuration{}.setTitle("Codeplay Mandelbrot Demo"),
      GLConfiguration{}.setFlags(GLConfiguration::Flag::QuietLog)},
    m_calc{WIDTH, HEIGHT},
    m_mesh{Magnum::MeshTools::compile(Magnum::Primitives::squareSolid(
      Magnum::Primitives::SquareFlag::TextureCoordinates))},
    m_shader{Magnum::Shaders::FlatGL2D::Configuration{}.setFlags(
      Magnum::Shaders::FlatGL2D::Flag::Textured |
      Magnum::Shaders::FlatGL2D::Flag::TextureTransformation)} {

    m_tex.setWrapping(Magnum::GL::SamplerWrapping::ClampToEdge)
          .setMagnificationFilter(Magnum::GL::SamplerFilter::Linear)
          .setMinificationFilter(Magnum::GL::SamplerFilter::Linear)
          .setStorage(1, Magnum::GL::textureFormat(PIXELFORMAT), {WIDTH,HEIGHT});
    m_shader.bindTexture(m_tex);
  }

  void tickEvent() override {
    // Transform coordinates from the ones used here - center point
    // and range - to the ones used in MandelbrotCalculator - min and max X, Y.
    double range_x = m_range * double(WIDTH) / double(HEIGHT);
    auto half_x = range_x / 2.0f;
    double min_x = m_ctr_x - half_x;
    double max_x = m_ctr_x + half_x;
    auto half_y = m_range / 2.0f;
    double min_y = m_ctr_y - half_y;
    double max_y = m_ctr_y + half_y;

    // Set new coordinates and recalculate the fractal
    m_calc.set_bounds(min_x, max_x, min_y, max_y);
    if (m_calc.supports_doubles()) {
      m_calc.calc<double>();
    } else {
      m_calc.calc<float>();
    }
  }

  void drawEvent() override {
    Magnum::GL::defaultFramebuffer.clear(
        Magnum::GL::FramebufferClear::Color |
        Magnum::GL::FramebufferClear::Depth);

    // Update GL texture with new calculation data
    m_calc.with_data([&](sycl::uchar4 const* data) {
      Magnum::ImageView2D img{
        PIXELFORMAT, {WIDTH,HEIGHT},
        Corrade::Containers::ArrayView{reinterpret_cast<const char*>(data), PIXELDATASIZE}};
      m_tex.setSubImage(0, {0,0}, img);
    });

    m_shader.draw(m_mesh);
    redraw();
    swapBuffers();
  }

  void mouseScrollEvent(MouseScrollEvent& event) override {
    // Zoom in or out on the plane
    auto inc = event.offset().y();
    if (inc > 0) {
      m_range *= 0.5f * inc;
    } else {
      m_range /= -0.5f * inc;
    }
  }

  void mouseMoveEvent(MouseMoveEvent& event) override {
    // Drag only if left button clicked
    if (!(event.buttons() & MouseMoveEvent::Button::Left)) {
      return;
    }
    // Calculate normalized coordinates
    auto x = event.position().x() / double(WIDTH);
    auto y = event.position().y() / double(HEIGHT);

    // Find the difference from last click
    auto dx = m_prev_mx - x;
    // y coords are reversed
    auto dy = y - m_prev_my;

    // If the difference is big enough, drag the center point
    // and with it the viewable part of the plane. The epsilon
    // is necessary to avoid noisy jumps
    constexpr double EPS = .1;
    if (dx < EPS && dx > -EPS) {
      m_ctr_x += dx * m_range;
    }
    if (dy < EPS && dy > -EPS) {
      m_ctr_y += dy * m_range * double(WIDTH) / double(HEIGHT);
    }

    m_prev_mx = x;
    m_prev_my = y;
  }
};

MAGNUM_APPLICATION_MAIN(MandelbrotApp)
