#include "speedSam.h"
#include "utils.h"

int main()
{
    // Option : Set the running Path
    std::string path = "";

    // Build the engines from onnx files
    SpeedSam Speedsam(path + "data/resnet18_image_encoder.onnx", path + "data/mobile_sam_mask_decoder.onnx");

    /*Segmentation examples */
    
    // Demo 1: Segment using a point
    segmentWithPoint(Speedsam, path + "assets/dog.jpg", path + "assets/dog_mask.jpg");
    
    // Demo 2: Segment using a bounding box
    segmentBbox(Speedsam, path + "assets/dogs.jpg", path + "assets/dogs_mask.jpg");

    return 0;
}
