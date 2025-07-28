from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from math import radians, sin, cos, sqrt, atan2

METERS_PER_FOOT = 0.3048
FEET_PER_MILE = 5280
EARTH_RADIUS_IN_METERS = 6371e3
METERS_PER_MILE = METERS_PER_FOOT * FEET_PER_MILE


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula a distância em milhas entre dois pontos geográficos
    usando a fórmula Haversine.
    """
    # Converter graus para radianos
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Diferenças das coordenadas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Fórmula Haversine
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance_meters = EARTH_RADIUS_IN_METERS * c

    # Converter para milhas
    return distance_meters / METERS_PER_MILE


def compute_distance(_spark: SparkSession, dataframe: DataFrame) -> DataFrame:
    """
    Adiciona coluna com distância calculada entre estações de início e fim.
    """
    # Registrar UDF (User Defined Function)
    distance_udf = udf(haversine_distance, DoubleType())

    # Adicionar coluna com distância calculada
    return dataframe.withColumn(
        "distance",
        distance_udf(
            dataframe["start_station_latitude"],
            dataframe["start_station_longitude"],
            dataframe["end_station_latitude"],
            dataframe["end_station_longitude"]
        )
    )


def run(spark: SparkSession, input_dataset_path: str, transformed_dataset_path: str) -> None:
    """
    Processa o dataset de entrada, calcula distâncias e salva o resultado.
    """
    input_dataset = spark.read.parquet(input_dataset_path)

    dataset_with_distances = compute_distance(spark, input_dataset)

    # Usar 'overwrite' em vez de 'append' para evitar duplicação
    dataset_with_distances.write.parquet(transformed_dataset_path, mode='overwrite')