# -*- coding: utf-8 -*-

import arcpy
from arcpy.sa import *
from arcpy.sa import BandArithmetic
import os
from pathlib import Path

class Toolbox(object):
    def __init__(self):
        """Esta herramienta permite generar un modelo de clasificación supervisada utilizando imágenes satelitales multibanda (por ejemplo, PlanetScope) y muestras de entrenamiento.

        Calcula automáticamente los índices espectrales NDVI y NDWI en función del número de bandas de la imagen (4 u 8), y los integra en una imagen compuesta. Luego, entrena un clasificador SVM (Support Vector Machine) usando las muestras proporcionadas y clasifica la imagen de entrada.

        Adicionalmente, evalúa la exactitud del resultado mediante una matriz de confusión generada a partir de puntos de validación.
        """
        self.label = "ImageClassifierIndex "
        self.alias = "ImageClassifierIndex "

        # List of tool classes associated with this toolbox
        self.tools = [Tool]


class Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "ImageClassifierIndex"
        self.description = "Clasificacion de imágenes satelitales con índices espectrales"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""

        imagen = arcpy.Parameter(
            displayName="Imagen satelital",
            name="imagen",
            datatype= ["GPRasterLayer"],
            direction="Input")
    
        muestras = arcpy.Parameter(
            displayName="Muestras",
            name="muestras",
            datatype= ["DEShapefile","GPFeatureLayer"],
            direction="Input")
        
        out_folder = arcpy.Parameter(
            displayName = "Carpeta de salida",
            name = "Output folder",
            datatype = "DEFolder",
            parameterType = "Required",
            direction = "Input")
    


        params = [imagen, muestras, out_folder]
        return params
        
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        imagen = parameters[0].valueAsText
        muestras = parameters[1].valueAsText
        out_folder = parameters[2].valueAsText
        

        out_training_feature_class = "out_trining"
        out_test_feature_class = "out_test"
        
        arcpy.CheckOutExtension("Spatial")
        desc = arcpy.Describe(imagen)
        bandas = desc.bandCount
        # Calcular NDVI según el número de bandas
        if bandas == 4:
            NDVI = BandArithmetic(imagen, "4 3", 1)
            NDWI = BandArithmetic(imagen, "2 4", 1)
            #nir = Raster(imagen + "/Band_4")  # NIR
            #red = Raster(imagen + "/Band_3")  # Red
            #green=Raster(imagen + "/Band_2")  # green
            arcpy.AddMessage("Imagen con 4 bandas detectada")

        elif bandas == 8:
            NDVI = BandArithmetic(imagen, "8 6", 1)
            NDWI = BandArithmetic(imagen, "4 8", 1)
            #nir = Raster(imagen + "/Band_8")  # NIR
            #red = Raster(imagen + "/Band_6")  # Red
            #green=Raster(imagen + "/Band_4")  # green
            arcpy.AddMessage("Imagen con 8 bandas detectada")

        else:
            arcpy.AddError(f"Número de bandas no soportado: {bandas}. Se esperaban 4 u 8 bandas.")
            raise SystemExit()

        # Cálculo de NDVI y NDWI
        #ndvi = (nir - red) / (nir + red)
        #ndwi= (green - nir) / (green + nir)
        ndvi_salida = os.path.join(out_folder, "ndvi.tif")
        ndwi_salida = os.path.join(out_folder, "ndwi.tif")
        #ndvi.save(ndvi_salida)
        #ndwi.save(ndwi_salida)
        
        NDVI.save(ndvi_salida)
        NDWI.save(ndwi_salida)
        
        # Crear imagen composite con los indices
        salida_composite = os.path.join(out_folder, "ndvi_ndwi_composite.tif")
        arcpy.management.CompositeBands(f"{ndvi_salida};{ndwi_salida}",salida_composite)

        arcpy.ga.SubsetFeatures(muestras, out_training_feature_class, out_test_feature_class, 70, "PERCENTAGE_OF_INPUT")
        
        
        path_imagen = Path(imagen)
        name_modelo = path_imagen.stem.split("_")[0]
        ruta_ecd = str(Path(out_folder) / f"{name_modelo}_modelo.ecd")
        # Entrenamiento del modelo SVM
        ecd =arcpy.ia.TrainSupportVectorMachineClassifier(
            in_raster=imagen,
            in_training_features=out_training_feature_class,
            out_classifier_definition=ruta_ecd,
            in_additional_raster=salida_composite,
            max_samples_per_class=1000,
            used_attributes="COLOR;MEAN",
            dimension_value_field="Classvalue"
        )
        
        # Clasificación de la imagen   
        ruta_clasificacion = str(Path(out_folder) / f"{name_modelo}_clasificacion.tif")
      
        
        classifiedraster = arcpy.sa.ClassifyRaster(imagen, ecd, salida_composite)
        classifiedraster.save(ruta_clasificacion)
        
           #AÑADIR EL RESULTADO AL MAPA
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        aprxMaps = aprx.listMaps()

        if aprxMaps:
            aprxMap = aprx.activeMap
            aprxMap.addDataFromPath(str(ruta_clasificacion))
            arcpy.AddMessage("El resultado se agregó al mapa, pero puede verificarlo en: {}".format(ruta_clasificacion))
        else:
            arcpy.AddMessage("Debe existir un mapa para poder visualizar el resultado, pero puede verificarlo en: {}".format(ruta_clasificacion))
   
        #MATRIZ DE CONFUSIÓN PARA EVALUAR EXACTITUD DEL RESULTADO DE LA CLASIFICACIÓN - Exactitud preliminar (antes de la edición)
        
        out_points = "in_memory/out_points"
        out_pointsSVM = "in_memory/out_pointsSVM"

        arcpy.sa.CreateAccuracyAssessmentPoints(out_test_feature_class, out_points)
        arcpy.ia.UpdateAccuracyAssessmentPoints(classifiedraster, out_points, out_pointsSVM, "GROUND_TRUTH")
        arcpy.sa.ComputeConfusionMatrix(out_pointsSVM, "MatrizConfusion")
        arcpy.AddMessage("Matriz de confusión creada exitosamente, revise la Geodatabase del proyecto") 
        return
