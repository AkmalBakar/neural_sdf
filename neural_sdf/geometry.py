import numpy as np
from decimal import *
import igl
import jax.numpy as jnp

class MeshSampler:
    """Sample points close to triangle mesh surface and in it's bounding box"""

    def __init__(self, vertices, faces, bounding_box, ratio=0.0, std=0.0):
        self._V = vertices
        self._F = faces
        self._BB = bounding_box

        # Check that bounding box encloses the mesh
        if not (np.all(vertices.min(axis=0) >= bounding_box[0]) and np.all(vertices.max(axis=0) <= bounding_box[1])):
            raise ValueError("Bounding box does not enclose the mesh")

        if ratio < 0 or ratio > 1:
            raise (ValueError("Ratio must be [0,1]"))

        self._ratio = ratio

        if std < 0:
            raise (ValueError("Std deviation must be non-negative"))

        self._std = std

        self._calculateFaceBins()

    def _calculateFaceBins(self):
        """Calculates and saves face area bins for sampling against"""
        vc = np.cross(
            self._V[self._F[:, 0], :] - self._V[self._F[:, 2], :],
            self._V[self._F[:, 1], :] - self._V[self._F[:, 2], :],
        )

        A = np.sqrt(np.sum(vc**2, 1))
        FA = A / np.sum(A)
        self._faceBins = np.concatenate(([0], np.cumsum(FA)))

    def _surfaceSamples(self, n):
        """Returns n points uniformly sampled from surface of mesh"""
        R = np.random.rand(n)  # generate number between [0,1]
        sampleFaceIdxs = np.array(np.digitize(R, self._faceBins)) - 1

        # barycentric coordinates for each face for each sample :)
        # random point within face for each sample
        r = np.random.rand(n, 2)
        A = self._V[self._F[sampleFaceIdxs, 0], :]
        B = self._V[self._F[sampleFaceIdxs, 1], :]
        C = self._V[self._F[sampleFaceIdxs, 2], :]
        P = (
            (1 - np.sqrt(r[:, 0:1])) * A
            + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
            + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
        )

        return P

    def _verticeSamples(self, n):
        """Returns n random vertices of mesh"""
        verts = np.random.choice(len(self._V), n)
        return self._V[verts]

    def _normalDist(self, V):
        """Returns normal distribution about each point V"""
        if self._std > 0.0:
            return np.random.normal(loc=V, scale=self._std)

        return V

    def _randomSamples(self, n):
        """Returns n random points in bounding box"""
        samples = np.random.uniform(self._BB[0], self._BB[1], (n, 3))
        return samples

    def _surfaceSamplesPerturbedNoBounds(self, n):
        """Returns surface samples perturbed with normal distribution, and thus, may fall outside the bounding box"""
        x = self._surfaceSamples(n)
        x_perturbed = np.random.normal(loc=x, scale=self._std)
        return x_perturbed

    def _surfaceSamplesPerturbed(self, n, max_tries=1000):
        """Returns surface samples perturbed with normal distribution while staying within the bounding box"""
        x_valid = np.empty((0, 3))  # Initialize with correct shape
        n_valid = 0
        num_tries = 0

        while n_valid < n:
            if num_tries >= max_tries:
                raise ValueError(
                    f"Reached maximum number of tries ({max_tries}) to sample points in the bounding box"
                )
            num_tries += 1

            n_remaining = n - n_valid
            x_test = self._surfaceSamplesPerturbedNoBounds(n_remaining)

            # Check which points are within the bounding box
            x_is_in = np.all((x_test >= self._BB[0]) & (x_test <= self._BB[1]), axis=1)

            # Append only valid samples
            x_valid = np.vstack([x_valid, x_test[x_is_in]])
            n_valid = len(x_valid)

        return x_valid[:n]

    def sample_points(self, n):
        """Returns n points according to point sampler settings"""

        nRandom = round(Decimal(n) * Decimal(self._ratio))
        nSurface = n - nRandom

        xRandom = self._randomSamples(nRandom)

        if nSurface > 0:
            xSurface = self._surfaceSamplesPerturbed(nSurface)

            if nRandom > 0:
                x = np.concatenate((xSurface, xRandom))
            else:
                x = xSurface
        else:
            x = xRandom

        np.random.shuffle(x)  # remove bias on order

        return x

    def sdf(self, pts):
        s, _, _ = igl.signed_distance(
            pts,
            self._V,
            self._F,
            sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
        )
        return s


from typing import Iterator, Tuple
import numpy as np
import jax.numpy as jnp

class SDFDataloader:
    """
    A data loader for Signed Distance Function (SDF) samples.

    This class manages the sampling, batching, and iteration over SDF data points.
    It supports resampling, shuffling, and dropping the last incomplete batch.

    Attributes:
        sampler: An object with sample_points and sdf methods for generating samples.
        num_samples (int): The total number of samples to generate.
        batch_size (int): The number of samples in each batch.
        resample_flag (bool): Whether to resample after each epoch.
        shuffle_flag (bool): Whether to shuffle the data after each epoch.
        drop_last (bool): Whether to drop the last batch if its size is less than batch_size.
        current_index (int): The current index in the sample array.
        x_samples (np.ndarray): Array of sampled points.
        sdf_samples (np.ndarray): Array of SDF values corresponding to x_samples.
    """

    def __init__(self, sampler: MeshSampler, num_samples: int, batch_size: int, 
                 shuffle: bool = True, resample: bool = False, drop_last: bool = True):
        """
        Initialize the SDFDataloader.

        Args:
            sampler: An object with sample_points and sdf methods for generating samples.
            num_samples (int): The total number of samples to generate.
            batch_size (int): The number of samples in each batch.
            shuffle (bool, optional): Whether to shuffle the data after each epoch. Defaults to True.
            resample (bool, optional): Whether to resample after each epoch. Defaults to False.
            drop_last (bool, optional): Whether to drop the last batch if its size is less than batch_size. Defaults to True.
        """
        self.sampler = sampler
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.resample_flag = resample
        self.shuffle_flag = shuffle
        self.drop_last = drop_last
        self.current_index = 0
        self.permutation = np.arange(self.num_samples)
        self.resample()

    def resample(self) -> None:
        """Resample points and their corresponding SDF values."""
        self.x_samples = self.sampler.sample_points(self.num_samples)
        self.sdf_samples = self.sampler.sdf(self.x_samples)
        self.current_index = 0
        self.permutation = np.arange(self.num_samples)  # Reset permutation

    def shuffle(self) -> None:
        """Shuffle the permutation array instead of the actual data."""
        np.random.shuffle(self.permutation)

    def reset(self) -> None:
        """Reset the dataloader's permutation to sequential order."""
        self.permutation = np.arange(self.num_samples)
        self.current_index = 0

    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Return the iterator object (self)."""
        return self

    def __next__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the next batch of samples.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the batch of x_samples and sdf_samples.

        Raises:
            StopIteration: When all samples have been iterated over.
        """
        end_of_epoch = self.current_index >= self.num_samples or (
            self.drop_last and self.num_samples - self.current_index < self.batch_size
        )
        if end_of_epoch:
            if self.resample_flag:
                self.resample()
            elif self.shuffle_flag:
                self.shuffle()
            self.current_index = 0
            raise StopIteration

        end_index = min(self.current_index + self.batch_size, self.num_samples)
        indices = self.permutation[self.current_index:end_index]
        x_batch = jnp.array(self.x_samples[indices])
        sdf_batch = jnp.array(self.sdf_samples[indices])

        self.current_index = end_index
        return x_batch, sdf_batch

    def __len__(self) -> int:
        """
        Get the number of batches in the dataloader.

        Returns:
            int: The number of batches.
        """
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    @property
    def domain_bounds(self) -> np.ndarray:
        """Get the domain bounds from the sampler.
        
        Returns:
            np.ndarray: Domain bounds with shape (2, 3)
        """
        return self.sampler._BB
