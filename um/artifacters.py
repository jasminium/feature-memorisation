from re import A
import numpy as np


class Artifacter():

    def __init__(self, artifact_offset, artifact, artifact_random):
        self.artifact_offset = artifact_offset
        self.artifact_offset_memory = artifact_offset
        self.artifact = artifact
        self.artifact_random = artifact_random

    def reset_offset(self):
        self.artifact_offset = self.artifact_offset_memory

    def augment_image(self, img, artifact):
        # add a text artifact to img
        """
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype("images/ArchivoBlack.otf", 20)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((0, 0), text, art_color, font=font)
        img = np.asarray(img)
        """
        img[self.artifact_offset:artifact.shape[0]+self.artifact_offset,
            self.artifact_offset:artifact.shape[1]+self.artifact_offset, :] = artifact

        return img

    def augment_images(self, imgs, artifact, feature_type=None):
        # add a text artifact to img
        """
        for i, img in enumerate(imgs):
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = ImageFont.truetype("images/ArchivoBlack.otf", 20)
            # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((0, 0), text, art_color, font=font)
            img = np.asarray(img)
            imgs[i] = img
            offset = 0
        """
        imgs[:, self.artifact_offset:artifact.shape[0]+self.artifact_offset, self.artifact_offset:artifact.shape[1] +
             self.artifact_offset, :] = np.stack([artifact]*imgs.shape[0], axis=0)
        return imgs



class ArtifacterRandom():

    def __init__(self, artifact_offset, artifact, artifact_random, n_imgs, transparent=False):
        self.artifact_offset = artifact_offset
        self.artifact_offset_memory = artifact_offset
        self.artifact = artifact
        self.artifact_random = artifact_random
        self.transparent = transparent
        self.n_imgs = n_imgs
        self.random_artifact_list = None

        self.update_random_artifacts()

    def update_random_artifacts(self):
        self.random_artifact_list = np.random.binomial(
            1, 0.5, size=(self.n_imgs, 5, 5, 1)).astype(np.float32)

    def reset_offset(self):
        self.artifact_offset = self.artifact_offset_memory

    def augment_image(self, img, artifact):
        # add a text artifact to img
        """
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype("images/ArchivoBlack.otf", 20)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((0, 0), text, art_color, font=font)
        img = np.asarray(img)
        """
        if self.transparent:

            # This only works with 3 channel images
            
            # make an artifact mask
            mask = np.zeros(img.shape[0:3])
            #mask = np.expand_dims(mask, axis=-1)
            mask[self.artifact_offset:artifact.shape[0]+self.artifact_offset,
                self.artifact_offset:artifact.shape[1]+self.artifact_offset, :] = artifact
            mask[mask != 0] = 1

            # boolean mask for np.place
            mask = (mask == 1)

            # create the artifact
            artifact_place = np.zeros(img.shape[0:3])
            #artifact_place = np.expand_dims(artifact_place, axis=-1)
            artifact_place[self.artifact_offset:artifact.shape[0]+self.artifact_offset,
                self.artifact_offset:artifact.shape[1]+self.artifact_offset, :] = artifact

            # update the image with the mask
            #np.place(img, mask, artifact_place)
            np.copyto(img, artifact_place, where=mask)

        else:
            img[self.artifact_offset:artifact.shape[0]+self.artifact_offset,
                self.artifact_offset:artifact.shape[1]+self.artifact_offset, :] = artifact

        return img

    def augment_images(self, imgs, artifact, feature_type=None):

        # this only works with 3 channel images.
        if self.transparent and feature_type=='feature':
            s = imgs.shape[-1]
            a = np.stack([artifact[:, :, 0]]*s, axis=-1)
            a = np.stack([a]*imgs.shape[0], axis=0)

            # make an artifact mask
            mask = np.zeros(imgs.shape[0:4], dtype=np.float32)
            mask[:, self.artifact_offset:artifact.shape[0]+self.artifact_offset, self.artifact_offset:artifact.shape[1] +
                self.artifact_offset, :] = a

            mask[mask != 0] = 1

            # boolean mask for np.place
            mask = (mask == 1)

            # create the artifact
            artifact_place = np.zeros(imgs.shape[0:4], dtype=np.float32)
            artifact_place[:, self.artifact_offset:artifact.shape[0]+self.artifact_offset, self.artifact_offset:artifact.shape[1] +
                self.artifact_offset, :] = a

            # update the image with the mask
            #np.place(imgs, mask, artifact_place)
            np.copyto(imgs, artifact_place, where=mask)

        elif feature_type == 'feature' or feature_type is None:
            imgs[:, self.artifact_offset:artifact.shape[0]+self.artifact_offset, self.artifact_offset:artifact.shape[1] +
                self.artifact_offset, :] = np.stack([artifact]*imgs.shape[0], axis=0)

        elif feature_type == 'random':
            imgs[:, self.artifact_offset:artifact.shape[0]+self.artifact_offset, self.artifact_offset:artifact.shape[1] +
                self.artifact_offset, :] = self.random_artifact_list
        else:
            raise Exception('Unknown feature type: {feature_type}')
        
        return imgs


class ArtifactorRandomNiNj(ArtifacterRandom):

    def __init__(self, artifact_offset, artifact, artifact_random, n_imgs, transparent=False, ni=None, nj=None, n_channels=1):
        self.ni = ni
        self.nj = nj
        self.n_channels = n_channels

        super().__init__(artifact_offset, artifact, artifact_random, n_imgs, transparent=transparent)

    def update_random_artifacts(self):
        self.random_artifact_list = np.random.binomial(
            1, 0.5, size=(self.nj, 5, 5, self.n_channels)).astype(np.float32)
        
        self.random_artifact_list = np.concatenate([self.random_artifact_list] * self.ni)

        print('update random artifact')

class ArtifactorRandomNiNjFixedSeed(ArtifacterRandom):

    def __init__(self, artifact_offset, artifact, artifact_random, n_imgs, transparent=False, ni=None, nj=None, n_channels=1):
        self.ni = ni
        self.nj = nj
        self.n_channels = n_channels

        super().__init__(artifact_offset, artifact, artifact_random, n_imgs, transparent=transparent)

    def update_random_artifacts(self):
        state = np.random.get_state()
        np.random.seed(1)
        self.random_artifact_list = np.random.binomial(
            1, 0.5, size=(self.nj, 5, 5, self.n_channels)).astype(np.float32)
        np.random.set_state(state)
        self.random_artifact_list = np.concatenate([self.random_artifact_list] * self.ni)
        print('update random artifact')

class ArtifactorRandomCifar(ArtifacterRandom):

    def __init__(self, artifact_offset, artifact, artifact_random, n_imgs, transparent=False):

        super().__init__(artifact_offset, artifact, artifact_random, n_imgs, transparent=transparent)

        self.random_artifact_list = np.random.binomial(
            1, 0.5, size=(n_imgs, 5, 5, 3)).astype(np.float32)


    
class ArtifacterRandomGen():

    def __init__(self, artifact_offset, artifact, artifact_random, n_imgs):
        self.artifact_offset = artifact_offset
        self.artifact_offset_memory = artifact_offset
        self.artifact = artifact
        self.artifact_random = artifact_random
        self.n_imgs = n_imgs

    def reset_offset(self):
        self.artifact_offset = self.artifact_offset_memory

    def augment_image(self, img, artifact):
        # add a text artifact to img
        img[self.artifact_offset:artifact.shape[0]+self.artifact_offset,
            self.artifact_offset:artifact.shape[1]+self.artifact_offset, :] = artifact

        return img

    def augment_images(self, imgs, artifact, feature_type=None):

        if feature_type == 'feature' or feature_type is None:
            imgs[:, self.artifact_offset:artifact.shape[0]+self.artifact_offset, self.artifact_offset:artifact.shape[1] +
                self.artifact_offset, :] = np.stack([artifact]*imgs.shape[0], axis=0)

        elif feature_type == 'random':
            imgs[:, self.artifact_offset:artifact.shape[0]+self.artifact_offset, self.artifact_offset:artifact.shape[1] +
                self.artifact_offset, :] = self.random_artifact_list = np.random.binomial(
                    1, 0.5, size=(self.n_imgs, 5, 5, 1)).astype(np.float32)
        else:
            raise Exception('Unknown feature type: {feature_type}')
        
        return imgs