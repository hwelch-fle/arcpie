from . import builtin as p
from .base import Controls, ParameterAttrs


class TextBox(p.GPString):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        super()._post_init(ctx)
        self.controlCLSID = Controls.STRING_TEXT_BOX


class NumberSlider(p.GPLong):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        super()._post_init(ctx)
        self.controlCLSID = Controls.NUMERIC_SLIDER


class EditableFeature(p.GPFeatureLayer):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        super()._post_init(ctx)
        self.controlCLSID = Controls.FEATURE_LAYER_CREATE


class HorizontalValueTable(p.GPValueTable):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        super()._post_init(ctx)
        self.controlCLSID = Controls.VALUE_TABLE_HORIZONTAL


class Done(p.GPBoolean):
    def __init__(self) -> None:
        super().__init__('Done')

    def _post_init(self, ctx: ParameterAttrs) -> None:
        super()._post_init(ctx)
        self.direction = 'Output'
        self.name = 'done'
        self.displayName = 'Done'
        self.parameterType = 'Derived'
        self.value = False


class Toggle(p.GPBoolean):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        if self.parameterType == 'Required':
            self.value = self.value if self.value is not None else False
        super()._post_init(ctx)


class String(p.GPString):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        if ctx.get('hidden'):
            self.datatype = 'GPStringHidden'
        super()._post_init(ctx)


class HiddenString(String):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        ctx['hidden'] = True
        super()._post_init(ctx)


class StringList(String):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        self.multiValue = True
        super()._post_init(ctx)


class FilePath(p.DEFile): ...


class MultiFilePath(p.DEFile):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        self.multiValue = True
        super()._post_init(ctx)


class Integer(p.GPLong): ...


class Double(p.GPDouble): ...


class FeatureLayer(p.GPFeatureLayer):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        if ctx.get('allow_create'):
            self.controlCLSID = Controls.FEATURE_LAYER_CREATE
        super()._post_init(ctx)


class FeatureLayerList(p.GPFeatureLayer):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        self.multiValue = True
        super()._post_init(ctx)


class Folder(p.DEFolder): ...


class SQLExpression(p.GPSQLExpression): ...


class FeatureDataset(p.DEFeatureDataset): ...


class Workspace(p.DEWorkspace): ...


class ValueTable(p.GPValueTable):
    def _post_init(self, ctx: ParameterAttrs) -> None:
        if 'defaults' not in ctx and 'default' in ctx:
            ctx['defaults'] = ctx['default']
        if 'filters' not in ctx and 'filter' in ctx:
            ctx['filters'] = ctx['filter']  # type: ignore

        cols = ctx.get('columns', {})
        flts = ctx.get('filters', {})
        dfts = ctx.get('defaults', [])

        if not cols:
            raise ValueError('Value Table needs columns!')

        self.columns = [[v, k] for k, v in cols.items()]
        for i, k in enumerate(cols):
            if k not in flts:
                continue
            self.filters[i].list = flts[k]
        if dfts:
            self.values = dfts

        # Don't allow default initializer to set these
        ctx.pop('defaults', None)
        ctx.pop('default', None)
        ctx.pop('filters', None)
        ctx.pop('filter', None)
        ctx.pop('columns', None)
        ctx.pop('values', None)

        super()._post_init(ctx)
