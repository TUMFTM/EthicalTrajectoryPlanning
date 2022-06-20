"""Extract the key outline coordinates of a shapely shape."""

import numpy as np
import shapely.geometry


def extract_polygon_outline(shapely_geometry: shapely.geometry) -> np.ndarray:
    """
    Extract the key outline coordinates of a shapely shape.

    Extract the key outline coordinates of a shapely shape
    (including multi-shapes like MultiPolygon). The following
    types ares supported:
        * Polygon:              The outline of the polygon is returned
        * MultiPolygon:         The outline of the largest polygon is returned
        * GeometryCollection:   The largest polygon in the set is returned,
                                if no polygon is present 'None' is returned
        * LineString:           'None' is returned, since the shape is
                                a line and does not host volume information

    For any other type, an error is raised.

    :param shapely_geometry:    shapely-geometry of interest
    :returns:
        * **polygon_outline** - outline coordinates in form of
                                a numpy array with columns x, y

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        09.10.2020

    """
    polygon_outline = None

    if shapely_geometry.geom_type == 'Polygon':
        if not shapely_geometry.is_empty:
            polygon_outline = shapely_geometry.exterior.coords.xy

    elif shapely_geometry.geom_type == 'MultiPolygon':
        # extract largest polygon
        polygon_outline = max(shapely_geometry, key=lambda a: a.area).\
            exterior.coords.xy

    elif shapely_geometry.geom_type == 'GeometryCollection':
        # extract polygons
        polygons = []
        for geometry in shapely_geometry.geoms:
            if (
                geometry.geom_type == 'Polygon' and
                not shapely_geometry.is_empty
            ):
                polygons.append(geometry)

        # extract largest polygon
        if polygons:
            polygon_outline = max(polygons, key=lambda a: a.area).\
                exterior.coords.xy

    elif shapely_geometry.geom_type == 'LineString':
        # if just line left, skip
        pass

    else:
        raise ValueError("Faced unsupported shape '" +
                         str(shapely_geometry.geom_type) + "'!")

    # convert to numpy array
    if polygon_outline is not None:
        polygon_outline = np.column_stack((
            polygon_outline[0],
            polygon_outline[1]))

    return polygon_outline
