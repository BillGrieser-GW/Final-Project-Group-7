# -*- coding: utf-8 -*-
"""
Find pixels key for removing a foreground digit by identifying background pixels
"""
import numpy as np
import torchvision.transforms as transforms 
import torch

to_PIL = transforms.Compose([transforms.ToPILImage()])
to_color_tensor = transforms.Compose([transforms.ToTensor()])

def light_dark_filter(threshold, candidates):
    
    filtered = []
    
     # Average the RGB and make a list of all averages to figure out light/dark
    # for the original image
    if len(candidates) > 0:
        values = np.array([c[2].mean().item() for c in candidates])
        val_median = np.percentile(values, 50)
        val_mean = values.mean()
    else:
        val_median, val_mean = (.5, .5)
        
    if val_median < val_mean:
        # We think the region we are matching is dark, cut off lighter candidates
        filtered = [c for c in candidates if c[2].mean().item() < (val_median + (values.std() * threshold))]
        print("Detecting dark target.")
    else:
        # We think the region we are matching is light; cut off darker candidates
        filtered = [c for c in candidates if c[2].mean().item() > (val_median - (values.std() * threshold))]
        print("Detecting light target.")
    
    return filtered


class KeyPixelFinder():
    
    def __init__(self, net, classes, device='cpu', 
                 lo_threshold_pct=0, hi_threshold_pct=8):
        self.net = net
        self.classes = classes
        self.device = device
        self.lo_threshold_pct = lo_threshold_pct
        self.hi_threshold_pct = hi_threshold_pct
       
    def get_using_grad_near_average(self, original_PIL_image, predicted_class=None, original_tensor=None):
        
        if original_tensor is None:
            original_tensor = self.net.get_transformer()(original_PIL_image)
        
        if predicted_class == None:
            outputs = self.net(original_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = int(predicted.cpu())
            
        # =============================================================================
        #  GRADS
        # =============================================================================
        other_grads = []
        grad_avg = torch.zeros((self.net.image_size[0], self.net.image_size[1]), device=self.device)
        weightsum = 0
       
        for idx in range(len(self.classes)):
            
            if original_tensor.grad is not None:
                original_tensor.grad.data.zero_()
                
            # Select the output we want grads for
            last_grad = [0] * len(self.classes)
            last_grad[idx] = 1
            
            outputs = self.net(original_tensor).to(device=self.device)
            
            outputs.backward(torch.Tensor(last_grad).to(device=self.device).view(1,-1), retain_graph=True)
            classgrads = original_tensor.grad[0,0].clone()
            
            # normalize the grads to a range 0 to 1
            classgrads = (classgrads - classgrads.min()) / (classgrads.max() - classgrads.min())
            
            if idx == predicted_class:
                pred_grads = classgrads.clone()
            else:
                other_grads.append(classgrads)
                grad_avg += classgrads
                weightsum += 1
               
        grad_avg = grad_avg / weightsum
        
        # Find the difference between the grads for the predicted class and
        # other grads for the same pixel
        thesegrads = abs(pred_grads - torch.tensor(grad_avg).type(torch.float)).cpu()
                
        # =============================================================================
        # IDENTIFY KEY PIXELS        
        # =============================================================================
        # Determine the grads who have the smallest differences from average
        threshold_lo = np.percentile(thesegrads.detach().numpy().flatten(), self.lo_threshold_pct)
        threshold_hi = np.percentile(thesegrads.detach().numpy().flatten(), self.hi_threshold_pct)
        quiet_pixels = 128 * ((thesegrads > threshold_lo) & (thesegrads < threshold_hi))
        
        return self._quiet_pixels_to_key_pixels(quiet_pixels, original_PIL_image, lambda x: light_dark_filter(0.20, x) )
    
    def get_using_grad_value(self, original_PIL_image, predicted_class=None, original_tensor=None):
        
        if original_tensor is None:
            original_tensor = self.net.get_transformer()(original_PIL_image)
        
        if predicted_class == None:
            outputs = self.net(original_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = int(predicted.cpu())
            
        # =============================================================================
        #  GRADS
        # =============================================================================
        if original_tensor.grad is not None:
            original_tensor.grad.data.zero_()
            
        # Select the output we want grads for
        last_grad = [0] * len(self.classes)
        last_grad[predicted_class] = 1
        
        outputs = self.net(original_tensor).to(device=self.device)
        
        outputs.backward(torch.Tensor(last_grad).to(device=self.device).view(1,-1), retain_graph=True)
        classgrads = original_tensor.grad[0,0].clone()
        
        # normalize the grads to a range 0 to 1
        classgrads = (classgrads - classgrads.min()) / (classgrads.max() - classgrads.min())
        
        # Find the difference between the grads for the predicted class and
        # other grads for the same pixel
        thesegrads = classgrads.cpu()
                
        # =============================================================================
        # IDENTIFY KEY PIXELS        
        # =============================================================================
        # Determine the grads who have the smallest relative values
        threshold_lo = np.percentile(thesegrads.detach().numpy().flatten(), self.lo_threshold_pct)
        threshold_hi = np.percentile(thesegrads.detach().numpy().flatten(), self.hi_threshold_pct)
        quiet_pixels = 128 * ((thesegrads > threshold_lo) & (thesegrads < threshold_hi))
        
        return self._quiet_pixels_to_key_pixels(quiet_pixels, original_PIL_image, lambda x: light_dark_filter(0.20, x) )
    
    def get_using_grid(self, original_PIL_image, predicted_class=None, original_tensor=None):
        
        quiet_pixels = torch.zeros((self.net.image_size[0], self.net.image_size[1]), dtype=torch.float)
        
         # Get back to a PIL Image matching the target size
        quiet_image = to_PIL(quiet_pixels.view(self.net.channels, self.net.image_size[0], 
                                               self.net.image_size[1])).resize((original_PIL_image.width, original_PIL_image.height)) 
        
        # Get the image pixels in a tensor where the last dimension is RGB (instead)
        # of the first dimension being the channel
        color_tensor = to_color_tensor(original_PIL_image).to(device=self.device).permute(1,2,0)

        # Selected pixels are potential traning data for the RBF net
        candidates = []
        for x in range(0, quiet_image.width, 3):
            for y in range(0, quiet_image.height, 3):
                candidates.append((x, y, color_tensor[y, x]))
                    
        # Apply filter if requested. Note that all candidates remain in the
        # quiet image
        key_pixels = light_dark_filter(0.1, candidates)
       
        # Add corners to the key pixels
        key_pixels.append((0, 0, color_tensor[0, 0]))
        key_pixels.append((original_PIL_image.width-1, 0, color_tensor[0, original_PIL_image.width-1]))
        key_pixels.append((original_PIL_image.width-1, original_PIL_image.height-1, color_tensor[original_PIL_image.height-1, original_PIL_image.width-1]))
        key_pixels.append((0, original_PIL_image.height-1, color_tensor[original_PIL_image.height-1, 0]))
        
        return key_pixels, quiet_image, candidates
        
                                                
    def _quiet_pixels_to_key_pixels(self, quiet_pixels, original_PIL_image, filter_=None, side_stride=0):
        """
        Helper function to build a list of key pixels and a image showing what
        pixels were considered and selected. This uses a heuristic to identify
        whether the background is light or dark and attempts to filter out 
        outliers.
        """
        # Get back to a PIL Image matching the target size
        quiet_image = to_PIL(quiet_pixels.view(self.net.channels, self.net.image_size[0], 
                                               self.net.image_size[1])).resize((original_PIL_image.width, original_PIL_image.height)) 
        
        # Get the image pixels in a tensor where the last dimension is RGB (instead)
        # of the first dimension being the channel
        color_tensor = to_color_tensor(original_PIL_image).to(device=self.device).permute(1,2,0)

        # Selected pixels are potential traning data for the RBF net
        candidates = []
        for x in range(quiet_image.width):
            for y in range(quiet_image.height):
                if quiet_image.getpixel((x,y)) > 0:
                    candidates.append((x, y, color_tensor[y, x]))
                    
        # Apply filter if requested. Note that all candidates remain in the
        # quiet image
        if filter_ != None:
            key_pixels = filter_(candidates)
        else:
            key_pixels = candidates
        
        # Add corners to the key pixels
        key_pixels.append((0, 0, color_tensor[0, 0]))
        key_pixels.append((original_PIL_image.width-1, 0, color_tensor[0, original_PIL_image.width-1]))
        key_pixels.append((original_PIL_image.width-1, original_PIL_image.height-1, color_tensor[original_PIL_image.height-1, original_PIL_image.width-1]))
        key_pixels.append((0, original_PIL_image.height-1, color_tensor[original_PIL_image.height-1, 0]))
        
        # Add sides to the key pixels
        if side_stride > 0:
            for x in range(0, original_PIL_image.width, side_stride):
                key_pixels.append((x, 0, color_tensor[0, x]))
                key_pixels.append((x, original_PIL_image.height-1, color_tensor[original_PIL_image.height-1, x]))
                
            for y in range(0, original_PIL_image.height, side_stride):
                key_pixels.append((0, y, color_tensor[y, 0])) 
                key_pixels.append((original_PIL_image.width-1, y, color_tensor[y, original_PIL_image.width-1])) 
                
        return key_pixels, quiet_image, candidates