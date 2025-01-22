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
 *    Application description for Fluid Simulation demo.
 *
 **************************************************************************/

#include "fluid.h"

#include <Corrade/Containers/StringStlView.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/ImageView.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Primitives/Square.h>
#include <Magnum/Shaders/FlatGL.h>
#include <Magnum/Trade/MeshData.h>

#include <sycl/sycl.hpp>

constexpr Magnum::PixelFormat PIXELFORMAT{Magnum::PixelFormat::RGBA8Unorm};

// Title bar text
constexpr std::string_view WINDOWTITLE{
    "Codeplay Fluid Simulation"
    " - Move mouse to add fluid"
    " - Press space to clear fluid"};

// Size of the fluid container edge (always square shaped).
constexpr int SIZE{300};

// Default scale at which fluid container is rendered.
constexpr int SCALE{3};

class FluidSimulationApp : public Magnum::Platform::Application {
 public:
  FluidSimulationApp(const Arguments& arguments)
      : Magnum::Platform::Application{arguments,
                                      Configuration{}.setTitle(WINDOWTITLE),
                                      GLConfiguration{}.setFlags(
                                          GLConfiguration::Flag::QuietLog)},
        size_{SIZE},
        fluid_{SIZE, 0.2f, 0.0f, 0.0000001f},
        mesh_{Magnum::MeshTools::compile(Magnum::Primitives::squareSolid(
            Magnum::Primitives::SquareFlag::TextureCoordinates))},
        shader_{Magnum::Shaders::FlatGL2D::Configuration{}.setFlags(
            Magnum::Shaders::FlatGL2D::Flag::Textured |
            Magnum::Shaders::FlatGL2D::Flag::TextureTransformation)} {
    // Set window size.
    setWindowSize({SIZE * SCALE, SIZE * SCALE});
    Magnum::GL::defaultFramebuffer.setViewport(
        {{0, 0}, {SIZE * SCALE, SIZE * SCALE}});

    texture_.setWrapping(Magnum::GL::SamplerWrapping::ClampToEdge)
        .setMagnificationFilter(Magnum::GL::SamplerFilter::Linear)
        .setMinificationFilter(Magnum::GL::SamplerFilter::Linear)
        .setStorage(1, Magnum::GL::textureFormat(PIXELFORMAT), {size_, size_});
    shader_.bindTexture(texture_);
  }

  // Called once per frame to update the fluid.
  void tickEvent() override final {
    // Check that mouse has moved.
    if (prev_x != 0 && prev_x != 0) {
      // Add density at mouse cursor location.
      auto x{static_cast<std::size_t>(prev_x * size_)};
      auto y{static_cast<std::size_t>(prev_y * size_)};
      fluid_.AddDensity(x, y, 400, 2);
    }

    // Fade overall dye levels slowly over time.
    fluid_.DecreaseDensity(0.99f);

    // Update fluid physics.
    fluid_.Update();
  }

  // Draws fluid to the screen.
  void drawEvent() override final {
    // Clear screen.
    Magnum::GL::defaultFramebuffer.clear(Magnum::GL::FramebufferClear::Color |
                                         Magnum::GL::FramebufferClear::Depth);

    // Update texture with pixel data array.
    fluid_.WithData([&](sycl::uchar4 const* data) {
      Magnum::ImageView2D img{
          PIXELFORMAT,
          {size_, size_},
          Corrade::Containers::ArrayView{
              reinterpret_cast<const char*>(data),
              size_ * size_ * Magnum::pixelFormatSize(PIXELFORMAT)}};
      texture_.setSubImage(0, {0, 0}, img);
    });

    // Draw texture to screen.
    shader_.draw(mesh_);
    redraw();
    swapBuffers();
  }

 private:
  void keyPressEvent(KeyEvent& event) override final {
    // Reset fluid container to empty if SPACE key is pressed.
    if (event.key() == KeyEvent::Key::Space) {
      fluid_.Reset();
    }
  }

  void mouseMoveEvent(MouseMoveEvent& event) override final {
    // Get mouse position as fraction of window size.
    auto x{event.position().x() / float(windowSize().x())};
    auto y{1.0f - event.position().y() / float(windowSize().y())};
    // Check that previous mouse position was not only zero (for initial value
    // to be set).
    if (prev_x != 0.0 || prev_y != 0.0) {
      // Get amount by which mouse has moved.
      auto amount_x{(x - prev_x) * size_};
      auto amount_y{(y - prev_y) * size_};
      // Add velocity and density in direction of mouse travel.
      // Multiplied by size because x and y are in 0.0 to 1.0 relative window
      // coordinates.
      auto current_x{static_cast<std::size_t>(x * size_)};
      auto current_y{static_cast<std::size_t>(y * size_)};
      fluid_.AddVelocity(current_x, current_y, amount_x, amount_y);
    }
    // Update previous mouse position.
    prev_x = x;
    prev_y = y;
  }

  // Square size of fluid container.
  int size_{0};

  // Store previous mouse positions.
  float prev_x{0};
  float prev_y{0};

  // Fluid container object.
  SYCLFluidContainer fluid_;

  // Fluid texture.
  Magnum::GL::Texture2D texture_;

  // Mesh and shader to draw the texture.
  Magnum::GL::Mesh mesh_;
  Magnum::Shaders::FlatGL2D shader_;
};

// Magnum app initialization.
MAGNUM_APPLICATION_MAIN(FluidSimulationApp)
