from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from uuid import uuid4

class EntidadBase(BaseModel):
    """Información genérica de todas las entidades"""
    nombre: str = Field(..., description="Nombre que representa a la entidad")
    descripcion: str = Field(..., description="Descripción identificativa de la entidad")

class Persona(EntidadBase):
    """Información de una persona"""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Identificador único de la persona")
    tipo: Literal['Persona']
    profesion: str = Field(..., description="Profesión u oficio de la persona")

class Empresa(EntidadBase):
    """Información de una empresa u organización"""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Identificador único de la empresa")
    tipo: Literal['Empresa']
    industria: Optional[str] = Field(None, description="Sector o industria de la empresa")

class CentroEducativo(EntidadBase):
    """Información de un centro educativo"""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Identificador único del centro educativo")
    tipo: Literal['CentroEducativo']
    localizacion: Optional[str] = Field(None, description="Ubicación o ciudad del centro educativo")

class Movimiento(EntidadBase):
    """Información de un movimiento o corriente de diseño/práctica"""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Identificador único del movimiento")
    tipo: Literal['Movimiento']
    categoria: Optional[str] = Field(
        None,
        description="Corriente o estilo (ej. minimalista, impresionista, art déco, etc.)"
    )

class Producto(EntidadBase):
    """
    Modelo que representa una entidad de tipo “Producto” y sus posibles subtipos.
    
    - tipo: indica que se trata de un producto.
    - subtipo: especifica si es un material, una técnica, un tipo genérico u otro.
    """
    id: str = Field(default_factory=lambda: str(uuid4()), description="Identificador único del producto")
    tipo: Literal['Producto'] = Field(
        'Producto',
        description="Constante que identifica la entidad como un producto."
    )
    subtipo: Optional[Literal['material', 'técnica', 'tipo', 'otro']] = Field(
        None,
        description="Subcategoría del producto. Puede ser 'material', 'técnica', 'tipo' u 'otro'."
    )

class Entidades(BaseModel):
    """Contenedor de todas las entidades extraídas"""
    personas: List[Persona] = Field(default_factory=list)
    empresas: List[Empresa] = Field(default_factory=list)
    centros_educativos: List[CentroEducativo] = Field(default_factory=list)
    movimientos: List[Movimiento] = Field(default_factory=list)
    productos: List[Producto] = Field(default_factory=list)

################################################
class Relacion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), description="Identificador único de la relación")
    entidad_origen: str = Field(..., description="Nombre que representa a la entidad de origen")
    entidad_destino: str = Field(..., description="Nombre que representa a la entidad de destino")
    descripcion_relacion: str = Field(..., description="explicación de por qué considera que la entidad origen y la entidad destino están relacionadas")
    fuerza_relacion: float = Field(..., description="puntuación numérica que indica la fuerza de la relación entre la entidad origen y la entidad destino")


class Relaciones(BaseModel):
    """Contenedor de todos los pares relacionados"""
    relaciones: List[Relacion] = Field(default_factory=list)


################################################
class OutputSchema(BaseModel):
    entidades: Entidades
    relaciones: Relaciones