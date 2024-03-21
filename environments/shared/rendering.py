"""
2D rendering of the level based foraging domain
"""

import os
import sys

import numpy as np
import math
import six
from gym import error

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size, hide_window=False):

        self.hide_window = hide_window
        self.rows, self.cols = world_size

        self.grid_size = 50
        self.icon_size = 20

        self.width = self.cols * self.grid_size + 1
        self.height = self.rows * self.grid_size + 1

        display = get_display(None)
        self.window = pyglet.window.Window(width=self.width, height=self.height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.window.set_visible(not hide_window)

        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = os.path.dirname(__file__)

        pyglet.resource.path = [os.path.join(script_dir, "icons")]
        pyglet.resource.reindex()

        self.img_active_apple = pyglet.resource.image("apple.png")
        self.img_foraged_apple = pyglet.resource.image("gray_apple.png")
        self.img_agent = pyglet.resource.image("agent.png")

    def close(self):
        if self.isopen:
            self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley))

    def render(self, env, return_rgb_array=False):

        glClearColor(0, 0, 0, 0)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_food(env, draw_id_badge=True)#not return_rgb_array)
        self._draw_players(env, draw_id_badge=True)#not return_rgb_array)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]

        self.window.flip()

        if return_rgb_array:
            return arr

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,
                        self.grid_size * r,
                        self.grid_size * self.cols,
                        self.grid_size * r,
                    ),
                ),
                ("c3B", (*_WHITE, *_WHITE)),
            )
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        self.grid_size * c,
                        0,
                        self.grid_size * c,
                        self.grid_size * self.rows,
                    ),
                ),
                ("c3B", (*_WHITE, *_WHITE)),
            )
        batch.draw()

    def _draw_food(self, env, draw_id_badge):

        apples = []
        batch = pyglet.graphics.Batch()

        # print(env.field)
        for apple_id, apple_location in enumerate(env.apple_locations):
            row, col = apple_location
            sprite = pyglet.sprite.Sprite(
                    self.img_active_apple if env.active_apples[apple_id] else self.img_foraged_apple,
                    self.grid_size * col,
                    self.height - self.grid_size * (row + 1),
                    batch=batch,
                )
            if not env.active_apples[apple_id]:
                sprite.update(scale=2)

            apples.append(
                sprite
            )

        for a in apples:
            a.update(scale=self.grid_size / a.width)

        batch.draw()

        for apple_id, apple_location in enumerate(env.apple_locations):
            if env.active_apples[apple_id]:
                row, col = apple_location
                if draw_id_badge: self._draw_id_badge(row, col, apple_id)
                self._draw_badge(row, col, env.apple_levels[apple_id])

    def _draw_players(self, env, draw_id_badge):
        players = []
        batch = pyglet.graphics.Batch()

        for agent_id, (row, col) in enumerate(env.agent_locations):
            players.append(
                pyglet.sprite.Sprite(
                    self.img_agent,
                    self.grid_size * col,
                    self.height - self.grid_size * (row + 1),
                    batch=batch,
                )
            )
        for p in players:
            p.update(scale=self.grid_size / p.width)
        batch.draw()
        for agent_id, level in enumerate(env.agent_levels):
            position = env.agent_locations[agent_id]
            self._draw_badge(*position, level)
            if draw_id_badge:
                self._draw_id_badge(*position, agent_id)

    def _draw_badge(self, row, col, level):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * self.grid_size + (3 / 4) * self.grid_size
        badge_y = self.height - self.grid_size * (row + 1) + (1 / 4) * self.grid_size

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_BLACK)
        circle.draw(GL_POLYGON)
        glColor3ub(*_WHITE)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(level),
            font_name="Times New Roman",
            font_size=12,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
        )
        label.draw()

    def _draw_id_badge(self, row, col, id):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * self.grid_size + (3 / 4) * self.grid_size
        badge_y = self.height - self.grid_size * (row + 1) + (1 / 4) * self.grid_size

        badge_x += -20
        badge_y += 18

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_BLACK)
        circle.draw(GL_POLYGON)
        glColor3ub(*_WHITE)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(id),
            font_name="Times New Roman",
            font_size=9,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
        )
        label.draw()


class NonEnvViewer(object):
    def __init__(self, world_size, icon_directory="../level_based_foraging/icons", hide_window=False):

        self.hide_window = hide_window
        self.rows, self.cols = world_size

        self.grid_size = 50
        self.icon_size = 20

        self.width = self.cols * self.grid_size + 1
        self.height = self.rows * self.grid_size + 1

        display = get_display(None)
        self.window = pyglet.window.Window(width=self.width, height=self.height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.window.set_visible(not hide_window)

        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = os.path.dirname(__file__)

        pyglet.resource.path = [os.path.join(script_dir, icon_directory)]
        pyglet.resource.reindex()

        self.render_level = "level" in icon_directory
        self.img_active_apple = pyglet.resource.image("apple.png")
        self.img_foraged_apple = pyglet.resource.image("gray_apple.png")
        self.img_agent = pyglet.resource.image("agent.png")

    def close(self):
        if self.isopen:
            self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley))

    def render(self, agent_locations, agent_levels, apple_locations, apple_levels, active_apples, return_rgb_array=False):

        glClearColor(0, 0, 0, 0)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_food(apple_locations, apple_levels, active_apples, draw_id_badge=True)#not return_rgb_array)
        self._draw_players(agent_locations, agent_levels, draw_id_badge=True)#not return_rgb_array)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]

        self.window.flip()

        if return_rgb_array:
            return arr

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,
                        self.grid_size * r,
                        self.grid_size * self.cols,
                        self.grid_size * r,
                    ),
                ),
                ("c3B", (*_WHITE, *_WHITE)),
            )
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        self.grid_size * c,
                        0,
                        self.grid_size * c,
                        self.grid_size * self.rows,
                    ),
                ),
                ("c3B", (*_WHITE, *_WHITE)),
            )
        batch.draw()

    def _draw_food(self, apple_locations, apple_levels, active_apples, draw_id_badge):

        apples = []
        batch = pyglet.graphics.Batch()

        # print(env.field)
        for apple_id, apple_location in enumerate(apple_locations):
            row, col = apple_location
            sprite = pyglet.sprite.Sprite(
                    self.img_active_apple if active_apples[apple_id] else self.img_foraged_apple,
                    self.grid_size * col,
                    self.height - self.grid_size * (row + 1),
                    batch=batch,
                )
            if not active_apples[apple_id]:
                sprite.update(scale=2)

            apples.append(
                sprite
            )

        for a in apples:
            a.update(scale=0.97 * (self.grid_size / a.width))

        batch.draw()

        for apple_id, apple_location in enumerate(apple_locations):
            if active_apples[apple_id]:
                row, col = apple_location
                if draw_id_badge: self._draw_id_badge(row, col, apple_id)
                if self.render_level: self._draw_badge(row, col, apple_levels[apple_id])

    def _draw_players(self, agent_locations, agent_levels, draw_id_badge):
        players = []
        batch = pyglet.graphics.Batch()

        for agent_id, (row, col) in enumerate(agent_locations):
            players.append(
                pyglet.sprite.Sprite(
                    self.img_agent,
                    self.grid_size * col,
                    self.height - self.grid_size * (row + 1),
                    batch=batch,
                )
            )
        for p in players:
            p.update(scale=0.97 * (self.grid_size / p.width))
        batch.draw()
        for agent_id, level in enumerate(agent_levels):
            position = agent_locations[agent_id]
            if self.render_level: self._draw_badge(*position, level)
            if draw_id_badge: self._draw_id_badge(*position, agent_id)

    def _draw_badge(self, row, col, level):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * self.grid_size + (3 / 4) * self.grid_size
        badge_y = self.height - self.grid_size * (row + 1) + (1 / 4) * self.grid_size

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_BLACK)
        circle.draw(GL_POLYGON)
        glColor3ub(*_WHITE)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(level),
            font_name="Times New Roman",
            font_size=12,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
        )
        label.draw()

    def _draw_id_badge(self, row, col, id):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * self.grid_size + (3 / 4) * self.grid_size
        badge_y = self.height - self.grid_size * (row + 1) + (1 / 4) * self.grid_size

        badge_x += -20
        badge_y += 18

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_BLACK)
        circle.draw(GL_POLYGON)
        glColor3ub(*_WHITE)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(id),
            font_name="Times New Roman",
            font_size=9,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
        )
        label.draw()
