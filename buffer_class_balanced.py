import torch
import numpy as np
from typing import Tuple
from torchvision import transforms


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer_class_balanced:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, num_classes, device,model,z_pred):
        self.buffer_size = buffer_size
        self.device = device
        self.num_classes = num_classes
        self.class_buffers = {i: [] for i in range(num_classes)}
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']
        self.z_pred = z_pred
        self.encoder = model

        self.class_prototypes = {i: torch.zeros(self.z_pred).to(device) for i in range(num_classes)}

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        self.attr_tensors = {}
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                self.attr_tensors[attr_str] = torch.zeros((self.buffer_size,
                                                           *attr.shape[1:]), dtype=typ, device=self.device)


    # balanced number of samples per class but this is randomly selected samples
    def add_data(self, examples, labels=None, logits=None, task_labels=None,encoder=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        self.encoder = encoder
        if not hasattr(self, 'attr_tensors'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            label = labels[i].item()
            class_buffer = self.class_buffers[label]
            index = reservoir(len(class_buffer), self.buffer_size // self.num_classes)
            if index >= 0:
                if index < len(class_buffer):
                    class_buffer[index] = (examples[i].to(self.device), labels[i].to(self.device),
                                           logits[i].to(self.device) if logits is not None else None,
                                           task_labels[i].to(self.device) if task_labels is not None else None)
                else:
                    class_buffer.append((examples[i].to(self.device), labels[i].to(self.device),
                                         logits[i].to(self.device) if logits is not None else None,
                                         task_labels[i].to(self.device) if task_labels is not None else None))
        
        self.update_prototypes()


    # balanced number of samples per class and encoder and metric based method
    def add_data1(self, examples, labels=None, logits=None, task_labels=None, encoder=None):
        """
        Adds the data to the memory buffer using a class-by-class strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param encoder: the model's encoder
        :return:
        """
        self.encoder = encoder
        if not hasattr(self, 'attr_tensors'):
            self.init_tensors(examples, labels, logits, task_labels)

        unique_labels = torch.unique(labels)

        for label in unique_labels:
            label = label.item()
            class_buffer = self.class_buffers[label]
            class_examples = examples[labels == label]
            class_labels = labels[labels == label]
            class_logits = logits[labels == label] if logits is not None else None
            class_task_labels = task_labels[labels == label] if task_labels is not None else None

            # Combine existing examples in the buffer with new examples of this class
            if len(class_buffer) > 0:
                existing_examples = torch.stack([item[0] for item in class_buffer]).to(self.device)
                combined_examples = torch.cat([existing_examples, class_examples], dim=0)

                with torch.no_grad():
                    combined_embeddings = self.encoder(combined_examples)

                # Exclude the new examples from the mean calculation for distance computation
                mean_embedding = combined_embeddings[:-class_examples.size(0)].mean(dim=0)
                distances = torch.norm(combined_embeddings - mean_embedding, dim=1)

                # Find the most distant sample index
                most_distant_indices = distances[:-class_examples.size(0)].argsort(descending=True).tolist()


            else:
                most_distant_indices = []

            if len(class_buffer) < self.buffer_size // self.num_classes:
                # Add new samples directly if buffer is not full
                for i in range(class_examples.size(0)):
                    class_buffer.append((
                        class_examples[i].to(self.device),
                        class_labels[i].to(self.device),
                        class_logits[i].to(self.device) if class_logits is not None else None,
                        class_task_labels[i].to(self.device) if class_task_labels is not None else None
                    ))
            else:
                # Replace the most distant samples with new ones if buffer is full
                num_to_replace = class_examples.size(0)
                for i in range(num_to_replace):
                    if most_distant_indices:
                        index_to_replace = most_distant_indices.pop(0)
                        class_buffer[index_to_replace] = (
                            class_examples[i].to(self.device),
                            class_labels[i].to(self.device),
                            class_logits[i].to(self.device) if class_logits is not None else None,
                            class_task_labels[i].to(self.device) if class_task_labels is not None else None
                        )

            self.update_prototypes()

    # interediate distanec based method ; this use mean embedding of curent available data

    def add_data2(self, examples, labels=None, logits=None, task_labels=None, encoder=None):
        """
        Adds the data to the memory buffer using a class-by-class strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param encoder: the model's encoder
        :return:
        """
        self.encoder = encoder
        if not hasattr(self, 'attr_tensors'):
            self.init_tensors(examples, labels, logits, task_labels)

        unique_labels = torch.unique(labels)

        for label in unique_labels:
            label = label.item()
            class_buffer = self.class_buffers[label]
            class_examples = examples[labels == label]
            class_labels = labels[labels == label]
            class_logits = logits[labels == label] if logits is not None else None
            class_task_labels = task_labels[labels == label] if task_labels is not None else None

            # Combine existing examples in the buffer with new examples of this class
            if len(class_buffer) > 0:
                existing_examples = torch.stack([item[0] for item in class_buffer]).to(self.device)
                combined_examples = torch.cat([existing_examples, class_examples], dim=0)

                with torch.no_grad():
                    combined_embeddings = self.encoder(combined_examples)

                # Compute mean embedding of the existing buffer examples
                mean_embedding = combined_embeddings[:-class_examples.size(0)].mean(dim=0)
                distances = torch.norm(combined_embeddings - mean_embedding, dim=1)

                # Determine distance thresholds
                num_existing_examples = len(existing_examples)
                distance_threshold_low = distances[:num_existing_examples].quantile(0.50)  # Lower quartile
                distance_threshold_high = distances[:num_existing_examples].quantile(0.99)  # Upper quartile

                # Select indices within the distance range
                within_range_indices = (distances >= distance_threshold_low) & (distances <= distance_threshold_high)
                valid_indices = within_range_indices.nonzero(as_tuple=True)[0]

                # Exclude the new examples from the selection process
                valid_indices = valid_indices[valid_indices < num_existing_examples]

            else:
                valid_indices = torch.arange(class_examples.size(0))  # All new examples are valid if buffer is empty

            if len(class_buffer) < self.buffer_size // self.num_classes:
                # Add new samples directly if buffer is not full
                for i in range(class_examples.size(0)):
                    class_buffer.append((
                        class_examples[i].to(self.device),
                        class_labels[i].to(self.device),
                        class_logits[i].to(self.device) if class_logits is not None else None,
                        class_task_labels[i].to(self.device) if class_task_labels is not None else None
                    ))
            else:
                # Replace existing examples with new examples within the valid distance range
                num_to_replace = class_examples.size(0)
                for i in range(num_to_replace):
                    if valid_indices.numel() > 0:
                        index_to_replace = valid_indices[i % valid_indices.numel()].item()  # Rotate if not enough valid indices
                        class_buffer[index_to_replace] = (
                            class_examples[i].to(self.device),
                            class_labels[i].to(self.device),
                            class_logits[i].to(self.device) if class_logits is not None else None,
                            class_task_labels[i].to(self.device) if class_task_labels is not None else None
                        )

            self.update_prototypes()

    # take mean proto from online learning and use that instead of current available data mean embedding
    def add_data_avg_proto(self, examples, labels=None, logits=None, task_labels=None, encoder=None, global_prototypes=None):
        """
        Adds the data to the memory buffer using a class-by-class strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param encoder: the model's encoder
        :return:
        """
        self.encoder = encoder
        if not hasattr(self, 'attr_tensors'):
            self.init_tensors(examples, labels, logits, task_labels)

        unique_labels = torch.unique(labels)

        for label in unique_labels:
            label = label.item()
            class_buffer = self.class_buffers[label]
            class_examples = examples[labels == label]
            class_labels = labels[labels == label]
            class_logits = logits[labels == label] if logits is not None else None
            class_task_labels = task_labels[labels == label] if task_labels is not None else None

            # Combine existing examples in the buffer with new examples of this class
            if len(class_buffer) > 0:
                existing_examples = torch.stack([item[0] for item in class_buffer]).to(self.device)
                combined_examples = torch.cat([existing_examples, class_examples], dim=0)

                with torch.no_grad():
                    combined_embeddings = self.encoder(combined_examples)

                # Compute mean embedding of the existing buffer examples
                #mean_embedding = combined_embeddings[:-class_examples.size(0)].mean(dim=0)

                mean_embedding = global_prototypes[label]
                distances = torch.norm(combined_embeddings - mean_embedding, dim=1)

                # Determine distance thresholds
                num_existing_examples = len(existing_examples)
                distance_threshold_low = distances[:num_existing_examples].quantile(0.0)  # Lower quartile
                distance_threshold_high = distances[:num_existing_examples].quantile(0.99)  # Upper quartile

                # Select indices within the distance range
                within_range_indices = (distances >= distance_threshold_low) & (distances <= distance_threshold_high)
                valid_indices = within_range_indices.nonzero(as_tuple=True)[0]

                # Exclude the new examples from the selection process
                valid_indices = valid_indices[valid_indices < num_existing_examples]

            else:
                valid_indices = torch.arange(class_examples.size(0))  # All new examples are valid if buffer is empty

            if len(class_buffer) < self.buffer_size // self.num_classes:
                # Add new samples directly if buffer is not full
                for i in range(class_examples.size(0)):
                    class_buffer.append((
                        class_examples[i].to(self.device),
                        class_labels[i].to(self.device),
                        class_logits[i].to(self.device) if class_logits is not None else None,
                        class_task_labels[i].to(self.device) if class_task_labels is not None else None
                    ))
            else:
                # Replace existing examples with new examples within the valid distance range
                num_to_replace = class_examples.size(0)
                for i in range(num_to_replace):
                    if valid_indices.numel() > 0:
                        index_to_replace = valid_indices[i % valid_indices.numel()].item()  # Rotate if not enough valid indices
                        class_buffer[index_to_replace] = (
                            class_examples[i].to(self.device),
                            class_labels[i].to(self.device),
                            class_logits[i].to(self.device) if class_logits is not None else None,
                            class_task_labels[i].to(self.device) if class_task_labels is not None else None
                        )

            self.update_prototypes()

    # mean prototype distance based method
    def add_data3(self, examples, labels=None, logits=None, task_labels=None, encoder=None,global_prototypes=None):
        """
        Adds data to the memory buffer based on mean prototype distance.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param encoder: the model's encoder
        :return:
        """
        self.encoder = encoder
        if not hasattr(self, 'attr_tensors'):
            self.init_tensors(examples, labels, logits, task_labels)

        unique_labels = torch.unique(labels)

        with torch.no_grad():
            # Compute embeddings for the incoming examples
            examples_embeddings = self.encoder(examples)

        for label in unique_labels:
            label = label.item()
            class_buffer = self.class_buffers[label]
            class_examples = examples[labels == label]
            class_labels = labels[labels == label]
            class_logits = logits[labels == label] if logits is not None else None
            class_task_labels = task_labels[labels == label] if task_labels is not None else None

            # Compute the prototype for the current class
            #prototype = self.class_prototypes[label]
            prototype = global_prototypes[label]

            # Compute embeddings for the new examples
            new_examples_embeddings = examples_embeddings[labels == label]

            # Calculate distances from the mean prototype
            distances = torch.norm(new_examples_embeddings - prototype, dim=1)

            # Sort distances and get indices of closest examples
            sorted_indices = torch.argsort(distances)

            # Calculate the number of examples to keep in the buffer
            num_to_keep = self.buffer_size // self.num_classes
            num_new_examples = class_examples.size(0)

            if len(class_buffer) < num_to_keep:
                # Add new samples directly if buffer is not full
                for i in range(num_new_examples):
                    class_buffer.append((
                        class_examples[i].to(self.device),
                        class_labels[i].to(self.device),
                        class_logits[i].to(self.device) if class_logits is not None else None,
                        class_task_labels[i].to(self.device) if class_task_labels is not None else None
                    ))
            else:
                # Replace the most distant samples with closest new examples
                num_to_replace = num_to_keep - len(class_buffer)
                for i in range(min(num_new_examples, num_to_replace)):
                    closest_index = sorted_indices[i].item()
                    index_to_replace = i % len(class_buffer)
                    class_buffer[index_to_replace] = (
                        class_examples[closest_index].to(self.device),
                        class_labels[closest_index].to(self.device),
                        class_logits[closest_index].to(self.device) if class_logits is not None else None,
                        class_task_labels[closest_index].to(self.device) if class_task_labels is not None else None
                    )

        self.update_prototypes()


    def get_data(self, size: int, transform: transforms = None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None:
            transform = lambda x: x

        samples_per_class = size // self.num_classes
        remainder = size % self.num_classes
        indices = []

        for class_id, class_buffer in self.class_buffers.items():
            class_size = len(class_buffer)
            num_samples = samples_per_class + (1 if class_id < remainder else 0)
            class_indices = np.random.choice(class_size, min(num_samples, class_size), replace=False)
            indices.extend((class_id, idx) for idx in class_indices)

        np.random.shuffle(indices)

        ret_tuple = ([], [], [], [])
        for class_id, idx in indices:
            example, label, logit, task_label = self.class_buffers[class_id][idx]
            ret_tuple[0].append(transform(example.cpu()).to(self.device))
            ret_tuple[1].append(label)
            if logit is not None:
                ret_tuple[2].append(logit)
            if task_label is not None:
                ret_tuple[3].append(task_label)

        ret_tuple = tuple(torch.stack(ret_attr) if len(ret_attr) > 0 else None for ret_attr in ret_tuple)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        return all(len(class_buffer) == 0 for class_buffer in self.class_buffers.values())

    def get_all_data(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None:
            transform = lambda x: x

        ret_tuple = ([], [], [], [])
        for class_buffer in self.class_buffers.values():
            for example, label, logit, task_label in class_buffer:
                ret_tuple[0].append(transform(example.cpu()).to(self.device))
                ret_tuple[1].append(label)
                if logit is not None:
                    ret_tuple[2].append(logit)
                if task_label is not None:
                    ret_tuple[3].append(task_label)

        ret_tuple = tuple(torch.stack(ret_attr) if len(ret_attr) > 0 else None for ret_attr in ret_tuple)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for class_id in range(self.num_classes):
            self.class_buffers[class_id] = []

    def update_prototypes(self):
        """
        Updates the prototypes for each class based on the current buffer.
        """
        for class_id, class_buffer in self.class_buffers.items():
            if len(class_buffer) > 0:
                examples = torch.stack([item[0] for item in class_buffer]).to(self.device)
                with torch.no_grad():
                    embeddings = self.encoder(examples)
                self.class_prototypes[class_id] = embeddings.mean(dim=0)
    # class prototypes for each class
    def get_class_prototypes(self,encoder) -> torch.Tensor:
        """
        Computes and returns the mean embeddings (class prototypes) for each label.
        :return: tensor of shape (num_classes, embedding_dim) containing the class prototypes
        """
        prototypes = []
        for class_id, class_buffer in self.class_buffers.items():
            if len(class_buffer) == 0:
                prototypes.append(torch.zeros(encoder.embedding_dim).to(self.device))
                continue

            examples = torch.stack([item[0] for item in class_buffer]).to(self.device)
            with torch.no_grad():
                embeddings = encoder(examples)
            mean_embedding = embeddings.mean(dim=0)
            prototypes.append(mean_embedding)
        
        return torch.stack(prototypes)