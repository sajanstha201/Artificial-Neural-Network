from django.shortcuts import render,redirect,HttpResponse
from src.pipeline.predict_pipeline import prediction
def sajan(request):
    return HttpResponse('<h1>sajan shresth</h1>')