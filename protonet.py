"""Implementation of prototypical networks for Omniglot."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # pylint: disable=unused-import
from torch.utils import tensorboard

import omniglot
import util  # pylint: disable=unused-import

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(DEVICE)
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 600


class ProtoNetNetwork(nn.Module):
    """Container for ProtoNet weights and image-to-latent computation."""

    def __init__(self):
        """Inits ProtoNetNetwork.

        The network consists of four convolutional blocks, each comprising a
        convolution layer, a batch normalization layer, ReLU activation, and 2x2
        max pooling for downsampling. There is an additional flattening
        operation at the end.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.
        """
        super().__init__()
        layers = []
        in_channels = NUM_INPUT_CHANNELS
        for _ in range(NUM_CONV_LAYERS):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    NUM_HIDDEN_CHANNELS,
                    (KERNEL_SIZE, KERNEL_SIZE),
                    padding='same'
                )
            )
            layers.append(nn.BatchNorm2d(NUM_HIDDEN_CHANNELS))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = NUM_HIDDEN_CHANNELS
        layers.append(nn.Flatten())
        self._layers = nn.Sequential(*layers)
        self.to(DEVICE)

    def forward(self, images):
        """Computes the latent representation of a batch of images.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)

        Returns:
            a Tensor containing a batch of latent representations
                shape (num_images, latents)
        """
        return self._layers(images)


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(self, learning_rate, log_dir):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
        """

        self._network = ProtoNetNetwork()
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _step(self, task_batch):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []                                               # batch = 16
        for task in task_batch:                                                 # task_batch 16 x 4 => each task size 4
            images_support, labels_support, images_query, labels_query = task   # support 5 x 1 x 28 x 28, query 75 x 1 x 28 x 28
            images_support = images_support.to(DEVICE)
            labels_support = labels_support.to(DEVICE)
            images_query = images_query.to(DEVICE)
            labels_query = labels_query.to(DEVICE)
            # ********************************************************
            # ******************* YOUR CODE HERE *********************
            # ********************************************************
            # TODO: finish implementing this method.
            # For a given task, compute the prototypes and the protonet loss.
            
            # Compute prototype from support examples

            # Get number of class
            #labels_sup_num = len(torch.unique(labels_support))
            #labels_que_num = len(torch.unique(labels_query))
            #print(labels_sup_num, labels_que_num) # 5, 5

            # Create new prototype
            #print(images_support.shape) #= N * K aka class * support
            # Group support vector with same label to the same group

            proto_dict = {}
            proto_count = {}
            for i in range(len(images_support)):
                each_sup = images_support[i]
                sup_label = labels_support[i]
                proto = self._network(torch.unsqueeze(each_sup, 0))
                # print(proto.shape) 1 * 64
                if sup_label.item() not in proto_dict.keys():
                    #print(sup_label.item())
                    proto_dict[sup_label.item()] = proto
                    proto_count[sup_label.item()] = 1
                else:
                    proto_dict[sup_label.item()] += proto
                    proto_count[sup_label.item()] += 1
            
            # Normalize centroids by number support vector
            for i in range(len(proto_dict.keys())):
                #print(proto_count[i])
                proto_dict[i] = proto_dict[i]/proto_count[i]

            # Get prediction for support set
            pred_support = []
            prob_support = {}
            support_assign = {}
            all_pred_sup = []
            for i in range(len(images_support)):
                # Embedding value
                proto = self._network(torch.unsqueeze(images_support[i], 0))
                min_dist = 0
                prob_sum_support = 0
                # Calculate distance with each prototype
                for each_proto in proto_dict.keys():
                    dist = torch.cdist(proto, proto_dict[each_proto])**2
                    prob_sum_support += torch.exp(-dist)
                    support_assign[each_proto] = torch.exp(-dist)
                    #print('Distance')
                    #print(dist)
                    #print(torch.exp(-dist))
                    prob_support[each_proto] = torch.exp(-dist)
                    if min_dist == 0:
                        min_dist = dist
                        pred_label = each_proto
                    elif dist < min_dist:
                        min_dist = dist
                        pred_label = each_proto
                
                support_assign[i] = min_dist
                all_pred_sup.append(each_proto)

                #print("prob_sum_support")
                #print(prob_sum_support) # should be 1 passed
                # assign probability
                pred_stack = []
                for prob in support_assign.keys():
                    prob_i = support_assign[prob]/prob_sum_support
                    #print(prob_i)
                    pred_stack.append(prob_i)
                    # check if sum = 1
                    #test += prob_i
                #print(test) # passed
                pred_support.append(pred_stack)

                
            #print('Boom')
            #print(prob_sum_support)

            pred_support = torch.Tensor(pred_support).cuda()
            #print(pred_support.shape)

            ################################################
            # Get accuracy of support set
            acc_support = util.score(pred_support, labels_support)
            accuracy_support_batch.append(acc_support)
            print(acc_support)
            ################################################

            # For each images in query set, put it through model and compare with each proto
            query_assign = {}
            prob_assign = {}
            batch_pred_stack = []
            all_pred = []
            for i in range(len(images_query)):
                each_que =  images_query[i]
                que_label = labels_query[i]
                # print(que_label) # 0 to 4
                embedding_dis = self._network(torch.unsqueeze(each_que, 0))
                #print(predict)
                # compare each image to cluster
                proto_prob_sum = 0
                min_dist = 0
                for each_c in proto_dict.keys():
                    c_centroid = proto_dict[each_c]
                    euclidena_dist = torch.cdist(c_centroid, embedding_dis)**2
                    proto_prob_sum += torch.exp(-euclidena_dist)
                    prob_assign[each_c] = torch.exp(-euclidena_dist)
                    # take centroid with least distance
                    if min_dist == 0:
                        min_dist = euclidena_dist
                        # assign centroid
                        min_value = each_c
                    elif euclidena_dist < min_dist:
                        min_dist = euclidena_dist
                        # assign centroid
                        min_value = each_c
                        # prediction
                        #all_pred.append(each_c)
                query_assign[i] = min_value
                all_pred.append(min_value)

                # assign unormalized log-probabilities
                #test = 0
                pred_stack = []
                for prob in prob_assign.keys():
                    prob_i = prob_assign[prob]/proto_prob_sum
                    # if prob_i > 0.9:
                    #     print(prob_i)
                    pred_stack.append(prob_i)
                    # check if sum = 1
                    #test += prob_i
                #print(test) # passed
                batch_pred_stack.append(pred_stack)

            #print(type(batch_pred_stack))
            #print(batch_pred_stack.dtype)
            #print(query_assign)
            #print(len(batch_pred_stack)) # 75 * 5 aka query x class

            # Use util.score to compute accuracies.
            batch_pred_stack = torch.Tensor(batch_pred_stack).cuda()
            #print(batch_pred_stack.shape)

            # Use F.cross_entropy to compute classification losses.

            #loss = F.cross_entropy(batch_pred_stack.type(torch.LongTensor), all_label.type(torch.LongTensor))
            #loss.backward()
            #print(loss)

            acc_query = util.score(batch_pred_stack, labels_query) # logit, label
            print(acc_query)
            # Make sure to populate loss_batch, accuracy_support_batch, and
            
            # accuracy_query_batch.
            accuracy_query_batch.append(acc_query)


            # ********************************************************
            # ******************* YOUR CODE HERE *********************
            # ********************************************************
        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch),
            np.mean(accuracy_query_batch)
        )

    def train(self, dataloader_train, dataloader_val, writer):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            loss, accuracy_support, accuracy_query = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            if i_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                    f'support accuracy: {accuracy_support.item():.3f}, '
                    f'query accuracy: {accuracy_query.item():.3f}'
                )
                writer.add_scalar('loss/train', loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/support',
                    accuracy_support.item(),
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/query',
                    accuracy_query.item(),
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    for val_task_batch in dataloader_val:
                        loss, accuracy_support, accuracy_query = (
                            self._step(val_task_batch)
                        )
                        losses.append(loss.item())
                        accuracies_support.append(accuracy_support)
                        accuracies_query.append(accuracy_query)
                    loss = np.mean(losses)
                    accuracy_support = np.mean(accuracies_support)
                    accuracy_query = np.mean(accuracies_query)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'support accuracy: {accuracy_support:.3f}, '
                    f'query accuracy: {accuracy_query:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/support',
                    accuracy_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/query',
                    accuracy_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/protonet/omniglot.way:{args.num_way}.support:{args.num_support}.query:{args.num_query}.lr:{args.learning_rate}.batch_size:{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    protonet = ProtoNet(args.learning_rate, log_dir)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = omniglot.get_omniglot_dataloader(
            'train',
            args.batch_size, # 16
            args.num_way, # 5
            args.num_support, # 1 
            args.num_query, # 15
            num_training_tasks # 
        )
        #print(len(dataloader_train)) # 15000
        dataloader_val = omniglot.get_omniglot_dataloader(
            'val',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4
        )
        protonet.train(
            dataloader_train,
            dataloader_val,
            writer
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS
        )
        protonet.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))

    main_args = parser.parse_args()
    main(main_args)
