from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
import os
from django.conf import settings
import traceback
from io import BytesIO
from huggingface_hub import hf_hub_download
import tensorflow as tf
import os

class ModelLoader:
    model_freshness = None
    model_disease = None

    @classmethod
    def load_freshness_model(cls):
        if cls.model_freshness is None:
            model_path = hf_hub_download(
                repo_id="saru03/Fishnet-Freshness",
                filename="fish-freshness.h5",
                cache_dir=os.path.join(settings.BASE_DIR, ".hf-cache")
            )
            cls.model_freshness = tf.keras.models.load_model(model_path)

    @classmethod
    def load_disease_model(cls):
        if cls.model_disease is None:
            model_path = hf_hub_download(
                repo_id="saru03/Fishnet-Disease",
                filename="disease-detection.h5",
                cache_dir=os.path.join(settings.BASE_DIR, ".hf-cache")
            )
            cls.model_disease = tf.keras.models.load_model(model_path)


class FreshnessCheckView(APIView):
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        ModelLoader.load_freshness_model()

        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'No image uploaded'}, status=400)

        class_names = ['Fresh', 'Highly Fresh', 'Not Fresh']
        try:
            image = Image.open(image_file)
            image = image.resize((224, 224))
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = ModelLoader.model_freshness.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_class = class_names[predicted_index]

            return Response({
                'freshness_score': predicted_class,
                'confidence': float(predictions[0][predicted_index])
            }, status=200)
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class DiseaseDetectionView(APIView):
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        ModelLoader.load_disease_model()

        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'No image uploaded'}, status=400)

        class_names = [
            'Bacterial Red disease',
            'Bacterial diseases - Aeromoniasis',
            'Bacterial gill disease',
            'Fungal diseases Saprolegniasis',
            'Healthy Fish',
            'Parasitic diseases',
            'Viral diseases White tail disease'
        ]
        try:
            img = tf.keras.utils.load_img(BytesIO(image_file.read()), target_size=(256, 256))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array,0)

            predictions = ModelLoader.model_disease.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_class = class_names[predicted_index]

            return Response({
                'disease_detection': predicted_class,
                'confidence': float(predictions[0][predicted_index])
            }, status=200)
        except Exception as e:
            traceback.print_exc()
            return Response({'error': str(e)}, status=500)
