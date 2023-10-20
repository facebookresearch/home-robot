from functools import cached_property

from dash_extensions.enrich import DashProxy


class DashComponent:
    def __init__(self, name):
        self.name = name

    def register_callbacks(self, app: DashProxy):
        pass

    @cached_property
    def layout(self):
        pass
