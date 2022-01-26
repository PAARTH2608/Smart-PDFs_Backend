import os
import cv2
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Image
from .serializers import ImageSerializer


class GetImagesView(APIView):
    def process(self, image):
        img = cv2.imread(image)
        print(img)


    def get(self, request, format=None):
        if Image.objects.all().exists():
            images = Image.objects.all()
            # GetImagesView().process(images[1].image)
            images = ImageSerializer(images, many=True)

            return Response(
                {'images': images.data},
                status=status.HTTP_200_OK
            )
        else:
            return Response(
                {'error': 'No images found'},
                status=status.HTTP_404_NOT_FOUND
            )

class ImageUploadView(APIView):

    # except : 
        # print("error")
    # def process(self, image):
    #     img = cv2.imread(image)
    #     print(img)


    def post(self, request):
        # try:
        data = self.request.data

        image = data['image']
        # print(image)
        # ImageUploadView().process(image)
        alt_text = data['alt_text']
        # ImageUploadView().process(str(image.read().decode()))
        Image.objects.create(
            image=image,
            alt=alt_text
        )
        return Response(
            {'success': 'Successfully uploaded image'},
            status=status.HTTP_201_CREATED
        )
    # except:
        return Response(
            {'error': 'Something went wrong when uploading image'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )