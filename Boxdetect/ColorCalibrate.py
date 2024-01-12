import cv2
import numpy as np

# def calHist_yuv(img):
#     # Compute histogram
#     img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#     hist_y = cv2.calcHist([img_yuv], [0], None, [256], [0, 256])
#     hist_u = cv2.calcHist([img_yuv], [1], None, [256], [0, 256])
#     hist_v = cv2.calcHist([img_yuv], [2], None, [256], [0, 256])
#     return hist_y, hist_u, hist_v

# def matchHistogram_yuv(hist_y, hist_u, hist_v, img):
#     img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#     cdf_y = hist_y.cumsum()
#     cdf_u = hist_u.cumsum()
#     cdf_v = hist_v.cumsum()

#     # Normalize CDF to the range [0, 255]
#     cdf_normalized_y = (cdf_y * 255) / cdf_y[-1]
#     cdf_normalized_u = (cdf_u * 255) / cdf_u[-1]
#     cdf_normalized_v = (cdf_v * 255) / cdf_v[-1]

#     # Map the intensity values using the CDF
#     equalized_channel_y = np.interp(img_yuv[:,:,0], range(256), cdf_normalized_y)
#     equalized_channel_u = np.interp(img_yuv[:,:,1], range(256), cdf_normalized_u)
#     equalized_channel_v = np.interp(img_yuv[:,:,2], range(256), cdf_normalized_v)

#     # Combine equalized channels into a 3D array
#     equalized_img_yuv = np.stack([equalized_channel_y, equalized_channel_u, equalized_channel_v], axis=-1)

#     # Convert back to BGR after converting to uint8
#     equalized_img_bgr = cv2.cvtColor(equalized_img_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)

#     return equalized_img_bgr






   
def calHist_y(img):
    # Compute histogram
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    hist_y, _ = np.histogram(img_yuv[:,:,0].flatten(), 256, [0, 256])
    return hist_y
def matchHistogram_y(hist_y, img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    cdf_y = hist_y.cumsum()

    # Normalize CDF to the range [0, 255]
    cdf_normalized_y = (cdf_y * 255) / cdf_y.max()


    # Map the intensity values using the CDF
    equalized_channel_y = cdf_normalized_y[img_yuv[:,:,0]]

    # Combine equalized channels into a 3D array
    img_yuv[:,:,0] = np.uint8(equalized_channel_y)

    # Convert back to BGR after converting to uint8
    equalized_img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return equalized_img_bgr

class CALIRBRATE():
    def __init__(self):
        self.bgr = 0
        self.matching = False
        self.target_cdf = None
        self.mapping = None
        self.tf = None


    
    def calHist_bgr(self,img):
        # Compute histogram
        self.hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
        self.hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        self.hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
        print("cal")
        # Normalize histograms to the range [0, thresh]
        self.bgr = 1

    def calHist_tf(self,img):
        # Load two images
        image1 = cv2.imread('image1.jpg') 
        image2 = cv2.imread('image2.jpg')

        # Calculate histograms
        hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

        # Normalize histograms
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)

        # Calculate cumulative histograms
        cumulative_hist1 = np.cumsum(hist1)
        cumulative_hist2 = np.cumsum(hist2)

        # Create a LUT
        lut = np.zeros(256, dtype=np.uint8)

        # Map pixel values based on cumulative histograms
        for i in range(256):
            lut[i] = np.argmin(np.abs(cumulative_hist1 - cumulative_hist2[i]))


    def matchHistogram_bgr(self, img):
        # print(self.bgr)
        if self.bgr == 0:
            return img
        else:
            img_bgr = img.copy()
            
            cdf_b = self.hist_b.cumsum()
            cdf_g = self.hist_g.cumsum()
            cdf_r = self.hist_r.cumsum()

            # Normalize CDF to the range [0, 255]
      
            cdf_normalized_b = (cdf_b * 255) / cdf_b[-1]
            cdf_normalized_g = (cdf_g * 255) / cdf_g[-1]
            cdf_normalized_r = (cdf_r * 255) / cdf_r[-1]

            # Map the intensity values using the CDF
            equalized_channel_b = np.interp(img_bgr[:,:,0], range(256), cdf_normalized_b)
            equalized_channel_g = np.interp(img_bgr[:,:,1], range(256), cdf_normalized_g)
            equalized_channel_r = np.interp(img_bgr[:,:,2], range(256), cdf_normalized_r)

            # Combine equalized channels into a 3D array
            equalized_img_bgr = np.stack([equalized_channel_b, equalized_channel_g, equalized_channel_r], axis=-1)

            return equalized_img_bgr.astype(np.uint8)
    def matchHistogram_tf(self, img):
        # print(self.bgr)
        if self.tf == 0:
            return img
        else:
             # Compute histogram
            hist_b = np.multiply(cv2.calcHist([img], [0], None, [256], [0, 256]),self.h_b)
            hist_g = np.multiply(cv2.calcHist([img], [1], None, [256], [0, 256]),self.h_g)
            hist_r = np.multiply(cv2.calcHist([img], [2], None, [256], [0, 256]),self.h_r)

        
            height, width = 1280, 720  # Adjust the size as needed
            result_image = np.zeros((height, width, 3), dtype=np.uint8)

            # Generate random pixel values based on the histograms
            result_image[:, :, 0] = np.random.choice(256, size=(height, width), p=hist_b.flatten() / hist_b.sum())
            result_image[:, :, 1] = np.random.choice(256, size=(height, width), p=hist_g.flatten() / hist_g.sum())
            result_image[:, :, 2] = np.random.choice(256, size=(height, width), p=hist_r.flatten() / hist_r.sum())


            return result_image.astype(np.uint8)
    # def histMatchCal(self, roi):
    #     try:
    #         # Load the reference image
    #         reference_image = cv2.imread('ref.png')

    #         if reference_image is None:
    #             raise Exception("Error loading reference image.")

    #         # Convert images to BGR color space
    #         reference_bgr = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    #         target_bgr = roi

    #         # Calculate histograms for the reference and target images
    #         reference_hist = cv2.calcHist([reference_bgr], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    #         target_hist = cv2.calcHist([target_bgr], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    #         # Normalize histograms
    #         cv2.normalize(reference_hist, reference_hist, 0, 1, cv2.NORM_MINMAX)
    #         cv2.normalize(target_hist, target_hist, 0, 1, cv2.NORM_MINMAX)

    #         # Calculate the cumulative distribution functions (CDF) for the histograms
    #         reference_cdf = reference_hist.cumsum()
    #         self.target_cdf = target_hist.cumsum()

    #         target_bgr = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    #         # Map the pixel values in the target image to the corresponding values in the reference image
    #         self.mapping = np.zeros(256, dtype=np.uint8)
    #         for i in range(256):
    #             print(f"Calibrating {int(i/256.0*100)} %                       ", end = '\r')
    #             diff = np.abs(self.target_cdf[i] - reference_cdf)
    #             min_diff_index = np.argmin(diff)
    #             self.mapping[i] = min_diff_index

    #         self.matching = True

    #     except Exception as e:
    #         print(f"Error in histMatchCal: {e}")
    #         self.matching = False

    # def histMatching(self, image):
    #     try:
    #         if self.matching:
    #             target_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #             # Apply the mapping to the target image
    #             matched_image = cv2.LUT(target_bgr, self.mapping)
    #             return matched_image
    #         else:
    #             return image

    #     except Exception as e:
    #         print(f"Error in histMatching: {e}")
    #         return image