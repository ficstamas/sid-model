import torch
from skimage import transform
import numpy as np
import torchvision.transforms.functional as tf
import numbers
import random
from PIL import Image
import Augmentor
import cairo
from math import pi
import cv2 as cv


class ComposeFANPortrait:
    def __init__(self, degree, p, magnitude, samples=1):
        self.degree = degree
        self.p = p
        self.samples = samples
        self.magnitude = magnitude

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        p = Augmentor.DataPipeline([[image, mask]])
        p.rotate(probability=self.p, max_left_rotation=self.degree, max_right_rotation=self.degree)
        p.random_distortion(probability=self.p, grid_height=5, grid_width=5, magnitude=1)
        p.skew_left_right(probability=self.p, magnitude=self.magnitude)
        # p.resize(probability=1, width=image.shape[0], height=image.shape[1], resample_filter="BILINEAR")
        sample, mask = p.sample(self.samples)[0]
        if sample.shape[0] != image.shape[0] or sample.shape[1] != image.shape[1]:
            out_image = np.zeros(image.shape, dtype=np.uint8)
            out_mask = np.zeros(image.shape, dtype=np.bool_)

            out_image[:sample.shape[0], :sample.shape[1], :] = sample[:, :, :]
            out_mask[:mask.shape[0], :mask.shape[1]] = mask[:, :]
            return out_image, out_mask
        return sample, mask


class ComposeFANInput:
    def __init__(self, size: tuple, portrait_scale=4, transforms=None):
        self.size = size
        self.portrait_scale = portrait_scale
        self.transform = transforms

    def __call__(self, img: np.ndarray, portrait: np.ndarray):

        mask1 = self.disk(img.shape[0], img.shape[1], 0.5, 0.5, 0.4)
        mask2 = self.disk(portrait.shape[0], portrait.shape[1], 0.5, 0.5, 0.4)

        dsize = (self.size[0], self.size[1])
        image1 = cv.resize(img, dsize)
        mask1 = cv.resize(mask1, dsize).astype(np.bool8)

        dsize = (self.size[0]//self.portrait_scale, self.size[0]//self.portrait_scale)
        image2 = cv.resize(portrait, dsize)
        mask2 = cv.resize(mask2, dsize).astype(np.bool8)

        image2 = np.array(Image.fromarray(image2).convert('LA'))[:, :, 0]
        image2 = np.stack((image2,) * 3, axis=-1)

        mask = mask1
        bg = image1

        random_position = np.random.randint(0, self.size[0]-self.size[0]//self.portrait_scale, [2])

        centered = random_position-self.size[0]//2
        if np.linalg.norm(centered, 2) < self.size[0]//2*0.8:
            x = centered / np.linalg.norm(centered, 2)
            z = x * 0.8 * (self.size[0]//2)
            random_position = z + (self.size[0]//2)
            if x[0] < 0:
                random_position[0] = random_position[0] - self.size[0]//self.portrait_scale//2
            if x[1] < 0:
                random_position[1] = random_position[1] - self.size[0]//self.portrait_scale//2
        random_position = random_position.astype(np.uint32)
        random_position[random_position + self.size[0]//self.portrait_scale > self.size[0]] = self.size[0]-self.size[0]//self.portrait_scale

        if self.transform:
            image2, mask2 = self.transform(image2, mask2)

        bg[random_position[0]:random_position[0]+image2.shape[0],
           random_position[1]:random_position[1]+image2.shape[1], :] = image2[:, :, :]

        mask[random_position[0]:random_position[0]+image2.shape[0],
             random_position[1]:random_position[1]+image2.shape[1]] |= mask2[:, :]

        return ToTensor()(bg / 255, mask)

    @staticmethod
    def disk(width, height, x, y, radius):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height) # A8
        cr = cairo.Context(surface)
        cr.scale(width, height)
        cr.set_source_rgba(0, 0, 0, 0)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        cr.set_source_rgba(1, 1, 1, 1)
        cr.arc(x, y, radius, 0, 2 * pi)
        cr.fill()
        buf = surface.get_data()
        data = np.ndarray(shape=(width, height), dtype=np.float32, buffer=buf) # np.dtype('>b1')
        return data


class Scale:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):

        numpy = False
        if type(img) is np.ndarray:
            img = Image.fromarray(img)
            numpy = True

        if type(mask) is np.ndarray:
            mask = Image.fromarray(mask)
            numpy = True

        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            oh = int(self.size * h)
            ow = int(self.size * w)
            rs_img = img.resize((ow, oh), Image.BILINEAR)
            rs_mask = mask.resize((ow, oh), Image.NEAREST)
            return np.array(rs_img) if numpy else rs_img, np.array(rs_mask) if numpy else rs_mask
        else:
            oh = int(self.size * h)
            ow = int(self.size * w)
            rs_img = img.resize((ow, oh), Image.BILINEAR)
            rs_mask = mask.resize((ow, oh), Image.NEAREST)
            return np.array(rs_img) if numpy else rs_img, np.array(rs_mask) if numpy else rs_mask


class RandomRotate:
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(
                img,
                translate=[0, 0],
                scale=1.0,
                angle=rotate_degree,
                resample=Image.BILINEAR,
                fillcolor=0,
                shear=[0.0],
            ),
            tf.affine(
                mask,
                translate=[0, 0],
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=0,
                shear=[0.0],
            ),
        )


class AdjustGamma:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation:
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            tf.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation)),
            mask,
        )


class AdjustHue:
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-self.hue, self.hue)), mask


class AdjustBrightness:
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf)), mask


class AdjustContrast:
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf)), mask


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomVerticallyFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask


class FreeScale:
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img


class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, mask):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float(), torch.from_numpy(mask)
