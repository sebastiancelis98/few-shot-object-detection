import os
import xml.etree.ElementTree as ET
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

def load_filtered_sdac_dataset(name, dirname, thing_classes):
    print(f"Loading {name} from {dirname}")
    data = []
    annotations_dir = os.path.join(dirname, "annots")
    images_dir = os.path.join(dirname, "images")

    is_shots = "shot" in name
    image_limit = -1
    if "image" in name:
        image_limit = int(name.split("image")[0].split("_")[-1])
        print(f"Image limit: {image_limit}")

    shots = name.split("_")[-1].split("shot")[0] if is_shots else -1
    if shots:
        shots = int(shots)

    shot_count = {}
    image_count = 0
    
    # Loop through all files in dirname 
    for filename in os.listdir(annotations_dir):
        if filename.endswith(".xml") and not filename.startswith("."):
            # Load xml file
            tree = ET.parse(os.path.join(annotations_dir, filename))
            root = tree.getroot()
            # Loop through all objects in xml file
            img_filename = filename.split(".")[0] + ".jpg"
            # Check if img_filename exists
            if not os.path.exists(os.path.join(images_dir, img_filename)):
                img_filename = root.find("path").text.split("/")[-1]
                if not os.path.exists(os.path.join(images_dir, img_filename)):
                    print(f"{img_filename} does not exist in {images_dir}")
                    continue
            
            if filename.split(".")[0] != img_filename.split(".")[0]:
                # Rename image file to match xml file
                new_img_filename = filename.split(".")[0] + ".jpg"
                os.rename(os.path.join(images_dir, img_filename), os.path.join(images_dir, new_img_filename))
                img_filename = new_img_filename
            
            for obj in root.findall("object"):
                obj_class = obj.find("name").text
                possible_classes = [thing_class for thing_class in thing_classes if obj_class.startswith(thing_class)]
                
                if possible_classes:
                    possible_classes.sort()
                    probable_class = possible_classes[-1]

                    if probable_class != obj_class:
                        # Set obj name to probable class
                        obj.find("name").text = probable_class
                        # Save obj file
                        tree.write(os.path.join(annotations_dir, filename))

                    # Get bounding box
                    bbox = obj.find("bndbox")

                    xmin = int(bbox.find("xmin").text)
                    ymin = int(bbox.find("ymin").text)
                    xmax = int(bbox.find("xmax").text)
                    ymax = int(bbox.find("ymax").text)

                    category_id = thing_classes.index(probable_class)

                    if is_shots and shot_count.get(category_id, 0) >= shots:
                        continue

                    shot_count[category_id] = shot_count.get(category_id, 0) + 1

                    bbox = {
                        "category_id": category_id,
                        "bbox": [xmin, ymin, xmax, ymax],
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }

                    annotation = {
                        "file_name": os.path.join(images_dir, img_filename),
                        "image_id": filename,
                        "width": int(root.findall("./size/width")[0].text),
                        "height": int(root.findall("./size/height")[0].text),
                        "id": len(data),
                        "annotations": [bbox],
                    }
                    data.append(annotation)

            image_count += 1

            if image_count >= image_limit and image_limit != -1:
                break
    return data

def register_meta_sdac(name, metadata, dirname, split, keepclasses):
    if keepclasses.startswith("base_novel"):
        thing_classes = metadata["thing_classes"]
    elif keepclasses.startswith("base"):
        thing_classes = metadata["base_classes"]
    elif keepclasses.startswith("novel"):
        thing_classes = metadata["novel_classes"]

    DatasetCatalog.register(
        name,
        lambda: load_filtered_sdac_dataset(
            name, dirname, thing_classes
        ),
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        split=split,
        base_classes=metadata["base_classes"],
        novel_classes=metadata["novel_classes"],
    )


if __name__ == "__main__":

    # Test output of sdac_loader, not used in the actual training/finetuning
    
    data = load_filtered_sdac_dataset('sdac_train_3images_1shot', 'datasets/sdac/small_img', [
        "ASR_Ansaugoeffnung", "ASR_Ansaugrauchmelder", "Ambulanz_Pendelleuchte_LED", "Aufbaurundleuchte_LED_Treppenhaus", "Ausschalter", "Ausschalter_beleuchtet", "Buero_Pendelleuchten_LED",
        "CEE_Drehstrom_Steckdose_16_32_A", "DB_Melder_in_Doppelboden_mit_Revisionsoeffnung_mind_40_x_40_cm", "Datenanschluss_einfach", "Datenanschluss_zweifach", "Downlight_LED_Einbauleuchten_Flur",
        "Downlight_LED_Einbauleuchten_Flur_2", "Einbaudownlights_LED_Konferenzraeume", "Elektroanschluss_allgemein_230V_Geraeteanschlussdose_230V", "Elektroanschluss_allgemein_400V_Geraeteanschlussdose_400V",
        "Fluchtwegpiktogramm", "Gegensprechanlage_mit_Video", "KFO_Kamera_mit_Festobjektiv", "NAM-Nichtautomatischer_Melder", "Notlichtbaustein_in_Allgemeinbeleuchtug_engebaut_Einbau_(1UG_und_2UG_Aufbauvariante)",
        "OKW_Multisensormelder_(Kombination_optischer_Rauch-Kahlenmonoxid_Waerme-Melder)", "OT_Multisensormelder_(Kombination_RMO_WMD)", "RMO_Rauchmelder_optischer", "ROA_Rauchmelder_optisch_mit_Signalgeber_(Sockelsirene)",
        "Ruf-und_Abstelltaster", "Scheinwerfer_fuer_Alarmbeleuchtung", "Schukosteckdose_einfach", "Schukosteckdose_mehrfach", "Serienschalter", "Sicherheitsleuchte_Wandeinbau", "Taster", "Taster_beleuchtet",
        "WLAN", "Wechselschalter", "ZD_Melder_in_Zwischendecke_mit_Revisionsoeffnung_mind_40_x_40_cm", "Zentrale_Sicherheitsbeleuchtung", "Zimmersignalleuchte", "Zugtaster"
        ])
    print(data)

    # Save data to json file
    with open('sdac_train_3images_1shot.json', 'w') as f:
        json.dump(data, f, indent=4)