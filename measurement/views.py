from django.conf import settings

import os
import os
import shutil

from django.shortcuts import render, redirect


from .modules.body_measurement import measure_body

def clear_folder(folder_path):
    """
    Delete all files and subfolders inside a folder.
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return

    for item in os.listdir(folder_path):

        item_path = os.path.join(folder_path, item)

        try:

            if os.path.isfile(item_path):
                os.remove(item_path)

            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        except Exception as e:
            print(e)

def home(request):

    if request.method == "POST":

        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        result_dir = os.path.join(settings.MEDIA_ROOT, "results")

        # Clear previous files
        clear_folder(upload_dir)
        clear_folder(result_dir)

        image = request.FILES["image"]

        image_path = os.path.join(
            upload_dir,
            image.name
        )

        with open(image_path, "wb+") as f:
            for chunk in image.chunks():
                f.write(chunk)

        # -------------------------
        # Run AI Measurement
        # -------------------------
        result = measure_body(image_path)

        # Save result for next page
        request.session["result"] = result

        return redirect("result")

    return render(request, "measurement/home.html")

def result(request):

    result = request.session.get("result")

    if result is None:
        return redirect("home")

    return render(
        request,
        "measurement/result.html",
        {
            "result": result
        }
    )