"""Custom exceptions for geo_spde package"""

class GeoSpdeError(Exception):
    """Base exception for geo_spde package"""
    pass

class CoordsError(GeoSpdeError):
    """Raised for coordinate processing errors"""
    pass

class MeshError(GeoSpdeError):
    """Raised for mesh generation errors"""
    pass

class MatrixError(GeoSpdeError):
    """Raised for matrix computation errors"""
    pass

class ParameterScaleError(GeoSpdeError):
    """Raised when SPDE parameters (kappa/tau) are poorly scaled for the mesh"""
    pass

class ConditioningError(GeoSpdeError):
    """Raised when numerical conditioning problems occur with precision matrix Q"""
    pass
