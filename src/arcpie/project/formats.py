from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

from arcpy import (
    _mp as mpt,
    mp,
)

if TYPE_CHECKING:
    from . import elements as elm
else:
    mpt = elm = type(
        '_mock', (object, ),
        {'__getattr__': lambda s, n: object}
    )()


__all__ = (
    'AIX',
    'BMP',
    'EMF',
    'EPS',
    'GIF',
    'JPEG',
    'PDF',
    'PNG',
    'SVG',
    'SVG',
    'TGA',
    'Format',
)


class FormatOpts(TypedDict, total=False):
    clipToElements: bool
    filePath: Path | str
    height: float
    width: float
    resolution: int


@dataclass
class Format[Formatter: mpt.ExportFormat = Any, Opts: Mapping[str, Any] = Any]:
    """Base Format class that contains all shared props and allows for isinstance checking for a ``Format`` object"""
    clipToElements: bool = False
    filePath: Path | str = ''
    height: float = 960
    width: float = 960
    resolution: int = 96

    @abstractmethod
    def update(self, mapping: Opts | None = None, /, **opts: Unpack[Opts]) -> None:  # type: ignore
        """Update the Format with the provided options.

        Args:
            mapping: An optional positional only dictionary (arg[0]) containing the update options
            **opts: Keyword overrides that will be applied to any supplied mapping arg
        """

    def _update(self, opts: Opts) -> None:
        for attr, val in opts.items():
            setattr(self, attr, val)

    @property
    def fmt(self) -> Formatter:
        fmt = mp.CreateExportFormat(type(self).__name__)  # type: ignore
        for attr, val in self.__dict__.items():
            if attr == 'filePath':
                setattr(fmt, attr, str(val))
            elif attr == 'imageCompression':
                fmt.setImageCompression(val)  # type: ignore
            elif attr == 'layersAndAttributes':
                fmt.setLayersAndAttributes(val)  # type: ignore
            elif attr == 'imageQuality':
                fmt.setImageQuality(val)  # type: ignore
            elif attr == 'colorMode':
                fmt.setColorMode(val)  # type: ignore
            elif attr == 'georeferenceMapFrame':
                if hasattr(val, 'elem') and type(val).__name__ == 'MapFrame':
                    val = val.elem
                if val is not None:
                    setattr(fmt, attr, val)
            elif hasattr(fmt, attr):
                setattr(fmt, attr, val)
            else:
                raise AttributeError(f'{type(self)} has no {attr} attribute')
        return fmt  # type: ignore


class AIXOpts(FormatOpts, total=False):
    convertMarkers: bool
    embedColorProfile: bool
    embedFonts: bool
    imageCompression: mpt.ImageCompression
    imageCompressionQuality: int
    imageQuality: mpt.ImageQuality
    removeLayoutBackground: bool


@dataclass
class AIX(Format[mpt.AIXFormat, AIXOpts]):
    """The ``AIXFormat`` object represents a collection of Adobe Illustrator Exchange (AIX)
    file properties that can be configured and used with the export method on the
    ``Layout``, ``MapFrame``, and ``MapView`` objects to create an output AIX file.

    [Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/aixformat-class.html)
    """

    convertMarkers: bool = False
    embedColorProfile: bool = True
    embedFonts: bool = True
    imageCompression: mpt.ImageCompression = 'ADAPTIVE'
    imageCompressionQuality: int = 80
    imageQuality: mpt.ImageQuality = 'BEST'
    removeLayoutBackground: bool = True

    def update(self, mapping: AIXOpts | None = None, /, **opts: Unpack[AIXOpts]) -> None:
        mapping = mapping or {}
        mapping.update(opts)
        return super()._update(mapping)


class BMPOpts(FormatOpts, total=False):
    colorMode: mpt.ColorMode
    georeferenceMapFrame: mpt.MapFrame | elm.MapFrame | None
    threshold: int
    worldFile: bool


@dataclass
class BMP(Format[mpt.BMPFormat, BMPOpts]):
    """The ``BMPFormat`` object represents a collection of Microsoft Windows bitmap (BMP)
    file properties that can be configured and used with the export method on the
    ``Layout``, ``MapFrame``, and ``MapView`` objects to create an output BMP file.

    [Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/bmpformat-class.html)
    """

    colorMode: mpt.ColorMode = '24-BIT_TRUE_COLOR'
    georeferenceMapFrame: mpt.MapFrame | elm.MapFrame | None = None
    threshold: int = 128
    worldFile: bool = False

    def update(self, mapping: BMPOpts | None = None, /, **opts: Unpack[BMPOpts]) -> None:
        mapping = mapping or {}
        mapping.update(opts)
        return super()._update(mapping)


class EMFOpts(FormatOpts, total=False):
    convertMarkers: bool
    imageQuality: mpt.ImageQuality
    outputAsImage: bool


@dataclass
class EMF(Format[mpt.EMFFormat, EMFOpts]):
    """The ``EMFFormat`` object represents a collection of Enhanced Metafile format (EMF)
    file properties that can be configured and used with the export method on the
    ``Layout``, ``MapFrame``, and ``MapView`` objects to create an output EMF file.

    [Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/emfformat-class.html)
    """

    convertMarkers: bool = False
    imageQuality: mpt.ImageQuality = 'BEST'
    outputAsImage: bool = False

    def update(self, mapping: EMFOpts | None = None, /, **opts: Unpack[EMFOpts]) -> None:
        mapping = mapping or {}
        mapping.update(opts)
        return super()._update(mapping)


class EPSOpts(FormatOpts, total=False):
    convertMarkers: bool
    embedFonts: bool
    imageCompression: mpt.ImageCompression
    imageQuality: mpt.ImageQuality
    outputAsImage: bool


@dataclass
class EPS(Format[mpt.EPSFormat, EPSOpts]):
    """The ``EPSFormat`` object represents a collection of Encapsulated PostScript (EPS)
    file properties that can be configured and used with the export method on the
    ``Layout``, ``MapFrame``, and ``MapView`` objects to create an output EPS file.

    [Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/epsformat-class.html)
    """

    convertMarkers: bool = False
    embedFonts: bool = True
    imageCompression: mpt.ImageCompression = 'DEFLATE'
    imageQuality: mpt.ImageQuality = 'BEST'
    outputAsImage: bool = False

    def update(self, mapping: EPSOpts | None = None, /, **opts: Unpack[EPSOpts]) -> None:
        mapping = mapping or {}
        mapping.update(opts)
        return super()._update(mapping)


class GIFOpts(FormatOpts, total=False):
    colorMode: mpt.ColorMode
    georeferenceMapFrame: mpt.MapFrame | elm.MapFrame | None
    threshold: int
    worldFile: bool


@dataclass
class GIF(Format[mpt.GIFFormat, GIFOpts]):
    """The ``GIFFormat`` object represents a collection of Graphic Interchange Format (GIF)
    file properties that can be configured and used with the export method on the
    ``Layout``, ``MapFrame``, and ``MapView`` objects to create an output GIF file.

    [Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/gifformat-class.html)
    """

    colorMode: mpt.ColorMode = '8-BIT_ADAPTIVE_PALETTE'
    georeferenceMapFrame: mpt.MapFrame | elm.MapFrame | None = None
    threshold: int = 128
    worldFile: bool = False

    def update(self, mapping: GIFOpts | None = None, /, **opts: Unpack[GIFOpts]) -> None:
        mapping = mapping or {}
        mapping.update(opts)
        return super()._update(mapping)


class JPEGOpts(FormatOpts, total=False):
    colorMode: mpt.ColorMode
    georeferenceMapFrame: mpt.MapFrame | elm.MapFrame | None
    imageCompressionQuality: int
    worldFile: bool


@dataclass
class JPEG(Format[mpt.JPEGFormat, JPEGOpts]):
    """The ``JPEGFormat`` object represents a collection of Joint Photographic Experts Group (JPEG)
    file properties that can be configured and used with the export method on the
    ``Layout``, ``MapFrame``, ``MapView``, ``MapSeries`` and ``BookmarkMapSeries`` objects to create an output JPG file.

    [Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/jpegformat-class.html)
    """

    colorMode: mpt.ColorMode = '24-BIT_TRUE_COLOR'
    georeferenceMapFrame: mpt.MapFrame | elm.MapFrame | None = None
    imageCompressionQuality: int = 80
    worldFile: bool = False

    def update(self, mapping: JPEGOpts | None = None, /, **opts: Unpack[JPEGOpts]) -> None:
        mapping = mapping or {}
        mapping.update(opts)
        return super()._update(mapping)


class PDFOpts(FormatOpts, total=False):
    author: str
    compressVectorGraphics: bool
    convertMarkers: bool
    embedColorProfile: bool
    embedFonts: bool
    georefInfo: bool
    imageCompression: mpt.ImageCompression
    imageCompressionQuality: int
    imageQuality: mpt.ImageQuality
    includeAccessibilityTags: bool
    includeNonVisibleMapLayers: bool
    keywords: str
    languageCode: str
    layersAndAttributes: mpt.LayerAttributes
    outputAsImage: bool
    rasterAsSingleTile: bool
    removeLayoutBackground: bool
    simulateOverprint: bool
    subject: str
    title: str


class _OldPDFSetting(TypedDict, total=False):
    resolution: int
    image_quality: mpt.ImageQuality
    compress_vector_graphics: bool
    image_compression: mpt.ImageCompression
    embed_fonts: bool
    layers_attributes: mpt.LayerAttributes
    georef_info: bool
    jpeg_compression_quality: int
    clip_to_elements: bool
    output_as_image: bool
    embed_color_profile: bool
    pdf_accessibility: bool
    keep_layout_background: bool
    convert_markers: bool
    simulate_overprint: bool


@dataclass
class PDF(Format[mpt.PDFFormat, PDFOpts]):
    """The ``PDFFormat`` object represents a collection of Portable Document Format (PDF)
    file properties that can be configured and used with the export method on the
    ``Layout``, ``MapFrame``, ``MapView``, ``MapSeries``, ``BookmarkMapSeries`` and ``Report``
    objects to create an output PDF file.

    [Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/pdfformat-class.html)
    """

    author: str = ''
    compressVectorGraphics: bool = True
    convertMarkers: bool = False
    embedColorProfile: bool = True
    embedFonts: bool = True
    georefInfo: bool = True
    imageCompression: mpt.ImageCompression = 'ADAPTIVE'
    imageCompressionQuality: int = 80
    imageQuality: mpt.ImageQuality = 'BEST'
    includeAccessibilityTags: bool = True
    includeNonVisibleMapLayers: bool = False
    keywords: str = ''
    languageCode: str = ''
    layersAndAttributes: mpt.LayerAttributes = 'LAYERS_ONLY'
    outputAsImage: bool = False
    rasterAsSingleTile: bool = False
    removeLayoutBackground: bool = False
    simulateOverprint: bool = False
    subject: str = ''
    title: str = ''

    def update(self, mapping: PDFOpts | None = None, /, **opts: Unpack[PDFOpts]) -> None:
        mapping = mapping or {}
        mapping.update(opts)
        return super()._update(mapping)

    @classmethod
    def convert(cls, setting: _OldPDFSetting) -> PDFOpts:
        arg_map = {
            'resolution': 'resolution',
            'image_quality': 'imageQuality',
            'compress_vector_graphics': 'compressVectorGraphics',
            'image_compression': 'imageCompression',
            'embed_fonts': 'embedFonts',
            'layers_attributes': 'layersAndAttributes',
            'georef_info': 'georefInfo',
            'jpeg_compression_quality': 'imageCompressionQuality',
            'clip_to_elements': 'clipToElements',
            'output_as_image': 'outputAsImage',
            'embed_color_profile': 'embedColorProfile',
            'pdf_accessibility': 'includeAccessibilityTags',
            'keep_layout_background': 'removeLayoutBackground',
            'convert_markers': 'convertMarkers',
            'simulate_overprint': 'simulateOverprint',
        }
        opts = cast(PDFOpts, {arg_map[key]: setting[key] for key in setting})
        if 'removeLayoutBackground' in opts:
            opts['removeLayoutBackground'] = not opts['removeLayoutBackground']
        return opts


class PNGOpts(FormatOpts, total=False):
    colorMode: mpt.ColorMode
    georeferenceMapFrame: mpt.MapFrame | elm.MapFrame | None
    threshold: int
    transparentBackground: bool
    worldFile: bool


@dataclass
class PNG(Format[mpt.PNGFormat, PNGOpts]):
    """The PNGFormat object represents a collection of Portable Network Graphics (PNG)
    file properties that can be configured and used with the export method on the
    Layout, MapFrame, MapView, MapSeries and BookmarkMapSeries objects to create an output PNG file.

    [Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/pngformat-class.html)
    """

    colorMode: mpt.ColorMode = '32-BIT_WITH_ALPHA'
    georeferenceMapFrame: mpt.MapFrame | elm.MapFrame | None = None
    threshold: int = 128
    transparentBackground: bool = False
    worldFile: bool = False

    def update(self, mapping: PNGOpts | None = None, /, **opts: Unpack[PNGOpts]) -> None:
        mapping = mapping or {}
        mapping.update(opts)
        return super()._update(mapping)


class SVGOpts(FormatOpts, total=False):
    compressToSVGZ: bool
    convertMarkers: bool
    embedFonts: bool
    imageQuality: mpt.ImageQuality
    includeNonVisibleMapLayers: bool
    outputAsImage: bool
    rasterAsSingleTile: bool


@dataclass
class SVG(Format[mpt.SVGFormat, SVGOpts]):
    """The ``SVGFormat`` object represents a collection of Scalable Vector Graphics (SVG)
    file properties that can be configured and used with the export method on the
    ``Layout``, ``MapFrame``, and ``MapView`` objects to create an output SVG or SVGZ file.

    [Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/svgformat-class.html)
    """

    compressToSVGZ: bool = False
    convertMarkers: bool = False
    embedFonts: bool = True
    imageQuality: mpt.ImageQuality = 'BEST'
    includeNonVisibleMapLayers: bool = False
    outputAsImage: bool = False
    rasterAsSingleTile: bool = False

    def update(self, mapping: SVGOpts | None = None, /, **opts: Unpack[SVGOpts]) -> None:
        mapping = mapping or {}
        mapping.update(opts)
        return super()._update(mapping)


class TGAOpts(FormatOpts, total=False):
    colorMode: mpt.ColorMode
    transparentBackground: bool


@dataclass
class TGA(Format[mpt.TGAFormat, TGAOpts]):
    """The ``TGAFormat`` object represents a collection of Truevision Graphics Adaptor (TGA)
    file properties that can be configured and used with the export method on the
    ``Layout``, ``MapFrame``, and ``MapView`` objects to create an output TGA file.

    [Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/tgaformat-class.html)
    """

    colorMode: mpt.ColorMode = '32-BIT_WITH_ALPHA'
    transparentBackground: bool = False

    def update(self, mapping: TGAOpts | None = None, /, **opts: Unpack[TGAOpts]) -> None:
        mapping = mapping or {}
        mapping.update(opts)
        return super()._update(mapping)


class TIFFOpts(FormatOpts, total=False):
    colorMode: mpt.ColorMode
    embedColorProfile: bool
    georeferenceMapFrame: mpt.MapFrame | elm.MapFrame | None
    geoTIFFTags: bool
    imageCompression: mpt.ImageCompression
    imageCompressionQuality: int
    threshold: int
    transparentBackground: bool
    worldFile: bool


@dataclass
class TIFF(Format[mpt.TIFFFormat, TIFFOpts]):
    """The ``TIFFFormat`` object represents a collection of Tagged Image File Format (TIFF)
    file properties that can be configured and used with the export method on the
    ``Layout``, ``MapFrame``, ``MapView``, ``MapSeries`` and ``BookmarkMapSeries``
    objects to create an output TIF file.

    [Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/tiffformat-class.html)
    """

    colorMode: mpt.ColorMode = '32-BIT_WITH_ALPHA'
    embedColorProfile: bool = True
    georeferenceMapFrame: mpt.MapFrame | elm.MapFrame | None = None
    geoTIFFTags: bool = False
    imageCompression: mpt.ImageCompression = 'LZW'
    imageCompressionQuality: int = 100
    threshold: int = 128
    transparentBackground: bool = False
    worldFile: bool = False

    def update(self, mapping: TIFFOpts | None = None, /, **opts: Unpack[TIFFOpts]) -> None:
        mapping = mapping or {}
        mapping.update(opts)
        return super()._update(mapping)
