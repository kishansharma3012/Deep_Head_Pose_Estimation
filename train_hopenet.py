import sys, os, argparse, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import datasets, hopenet, utils
import torch.utils.model_zoo as model_zoo
torch.manual_seed(0)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--Model_num', dest='Model_num', help='Model 0 - HopeNet, Model 1- HopenNet_featstack',
          default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=2, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--output_dir', dest='output_dir', help='Path to output_dir',
          default='', type=str)
    parser.add_argument('--Loss_func', dest='Loss_func', help='Loss Function',
          default='MSE', type=str)
    parser.add_argument('--num_bins', dest='num_bins', help='Number of Bins',
          default=66, type=int)
    parser.add_argument('--class_weight', dest='class_weight', help='class weight',
          default='', type=str)
    parser.add_argument('--debug', dest='debug', help='debug',
          default='False', type=str)

    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

def log_results( train_loss,  val_loss, Log_dir, plotname, i ):

    """
    Function - Logging and plotting of training and validation loss
    
    Input- 
    train_loss : Training loss list
    val_loss   : Validation loss list
    Log_dir    : Output directory path
    plotname   : name of the output plot 
    i          : Figure number of matplotlib
    """
    if not os.path.exists(Log_dir):
        os.mkdir(Log_dir)    

    Error_file  = open(Log_dir + '/' + plotname + '_Summary.txt', "w")
    Error_file.write('Train_loss: ' + str(train_loss))
    Error_file.write('\n val_loss: '+  str(val_loss))
    Error_file.close()

    plt.figure(i+10, figsize=(15, 10))
    plt.plot(range(len(train_loss)), train_loss, 'r', label = 'train')
    plt.plot(range(len(val_loss)), val_loss, 'g', label = 'val')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Train & Validation ' + plotname)
    plt.savefig(Log_dir + '/' + plotname + '.png')
    plt.close()

def reset_loss_history():
    loss_cross_yaw = []
    loss_cross_pitch = []
    loss_cross_roll = []
    
    loss_MSE_yaw = []
    loss_MSE_pitch = []
    loss_MSE_roll = []
    
    loss_yaw = []
    loss_pitch = []
    loss_roll = []
    return loss_cross_yaw, loss_cross_pitch, loss_cross_roll, loss_MSE_yaw, loss_MSE_pitch, loss_MSE_roll,\
        loss_yaw, loss_pitch, loss_roll

def update_loss_history(losses_lists, l_cross_yaw, l_cross_pitch, l_cross_roll, l_MSE_yaw, l_MSE_pitch, l_MSE_roll,\
            l_yaw, l_pitch, l_roll):
    losses_lists[0].append(l_cross_yaw)
    losses_lists[1].append(l_cross_pitch)
    losses_lists[2].append(l_cross_roll)
    losses_lists[3].append(l_MSE_yaw)
    losses_lists[4].append(l_MSE_pitch)
    losses_lists[5].append(l_MSE_roll)
    losses_lists[6].append(l_yaw)
    losses_lists[7].append(l_yaw)
    losses_lists[8].append(l_yaw)
    
    return losses_lists
        
if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    num_bins = args.num_bins    
    output_dir = os.path.join(os.getcwd(),args.output_dir) 
    alpha = args.alpha
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)    

    ##########################################
    ##           Model Selection            ##
    ##########################################

    if args.Model_num == 0:
        model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins)
    elif args.Model_num == 1:
        model = hopenet.Hopenet_FeatStack(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins)
    
    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    
    ##########################################
    ##           Dataset Loading            ##
    ##########################################
    print('Loading data.')
    transformations = transforms.Compose([transforms.Scale(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        train_filename_list = os.path.join(args.filename_list ,'train_filename_all.npy')
        val_filename_list = os.path.join(args.filename_list ,'val_filename_all.npy')
        pose_dataset_train = datasets.Pose_300W_LP(args.data_dir, num_bins, train_filename_list, transformations, args.debug)
        pose_dataset_val = datasets.Pose_300W_LP(args.data_dir, num_bins, val_filename_list, transformations, args.debug)
    elif args.dataset == 'AFLW2000':
        test_filename_list = os.path.join(args.filename_list ,'test_filename.npy')
        pose_dataset = datasets.AFLW2000(args.data_dir, num_bins, args.filename_list, transformations)
    else:
        print('Error: not a valid dataset name')
        sys.exit()

    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset_train,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=pose_dataset_val,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4)
    
    ## Handling Class Imbalance
    yaw_l = []
    pitch_l = []
    roll_l = []
    for i in range(len(pose_dataset_train)):
        y,p,r = pose_dataset_train[i][2]
        y_l, p_l, r_l = pose_dataset_train[i][1]
        if y_l < num_bins:
            yaw_l.append(y_l)
        if p_l < num_bins:
            pitch_l.append(p_l)
        if r_l < num_bins:
            roll_l.append(r_l)
    yaw_l = np.array(yaw_l)
    pitch_l = np.array(pitch_l)
    roll_l = np.array(roll_l)
    
    if args.class_weight == 'balanced':
        class_weights_yaw = np.zeros((num_bins,))
        class_weights_yaw[np.unique(yaw_l)] = compute_class_weight('balanced', np.unique(yaw_l), yaw_l)
        class_weights_yaw =  torch.FloatTensor(class_weights_yaw).cuda(gpu)
        
        class_weights_pitch = np.zeros((num_bins,))
        class_weights_pitch[np.unique(pitch_l)] = compute_class_weight('balanced', np.unique(pitch_l), pitch_l)
        class_weights_pitch =  torch.FloatTensor(class_weights_pitch).cuda(gpu)
        
        class_weights_roll = np.zeros((num_bins,))
        class_weights_roll[np.unique(roll_l)] = compute_class_weight('balanced', np.unique(roll_l), roll_l)
        class_weights_roll =  torch.FloatTensor(class_weights_roll).cuda(gpu)

    else:
        class_weights_yaw = np.ones((num_bins,))
        class_weights_yaw =  torch.FloatTensor(class_weights_yaw).cuda(gpu)

        class_weights_pitch = np.ones((num_bins,))
        class_weights_pitch =  torch.FloatTensor(class_weights_pitch).cuda(gpu)

        class_weights_roll = np.ones((num_bins,))
        class_weights_roll =  torch.FloatTensor(class_weights_roll).cuda(gpu)

    ##########################################
    ##           Loss Functions             ##
    ##########################################

    criterion_yaw = nn.CrossEntropyLoss(weight = class_weights_yaw).cuda(gpu)
    criterion_pitch = nn.CrossEntropyLoss(weight = class_weights_pitch).cuda(gpu)
    criterion_roll = nn.CrossEntropyLoss(weight = class_weights_roll).cuda(gpu)
    
    if args.Loss_func == 'MSE':
        reg_criterion = nn.MSELoss().cuda(gpu)
    elif args.Loss_func == 'MAE':
        reg_criterion = nn.L1Loss().cuda(gpu)
    
    softmax = nn.Softmax().cuda(gpu)
    idx_tensor = [idx for idx in range(num_bins)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)
    optimizer = torch.optim.Adam( model.parameters(),
                                   lr = args.lr)
    
    
    ##########################################
    ##              Training                ##
    ##########################################
    print('Ready to train network.')
    train_loss_history_lists = reset_loss_history()
    val_loss_history_lists = reset_loss_history()
    
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(num_epochs):

        train_loss_lists = reset_loss_history()
        # Training Network
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)

            if  (labels < 0 ).any() or (labels >= num_bins).any():
                continue
            
            # Binned labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)
            
            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_cross_yaw = criterion_yaw(yaw, label_yaw)
            loss_cross_pitch = criterion_pitch(pitch, label_pitch)
            loss_cross_roll = criterion_roll(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)
            
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw = loss_cross_yaw + alpha * loss_reg_yaw
            loss_pitch = loss_cross_pitch + alpha * loss_reg_pitch
            loss_roll = loss_cross_roll + alpha * loss_reg_roll

            train_loss_lists = update_loss_history(train_loss_lists, loss_cross_yaw.item(), loss_cross_pitch.item(), loss_cross_roll.item(), \
                                loss_reg_yaw.item(), loss_reg_pitch.item(), loss_reg_roll.item(),\
                                loss_yaw.item(), loss_pitch.item(), loss_roll.item())

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = [torch.ones(1).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()
            
            if (i+1) % 1000 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                       %(epoch+1, num_epochs, i+1, len(pose_dataset_train)//batch_size, np.mean(train_loss_lists[6]), np.mean(train_loss_lists[7]), np.mean(train_loss_lists[8])))
        
        train_loss_history_lists = update_loss_history(train_loss_history_lists, np.mean(train_loss_lists[0]), np.mean(train_loss_lists[1]),\
                            np.mean(train_loss_lists[2]), np.mean(train_loss_lists[3]), np.mean(train_loss_lists[4]), np.mean(train_loss_lists[5]),\
                            np.mean(train_loss_lists[6]), np.mean(train_loss_lists[7]), np.mean(train_loss_lists[8]))

        # Validation
        val_loss_lists = reset_loss_history()
        for i, (images, labels, cont_labels, name) in enumerate(val_loader):
            images = Variable(images).cuda(gpu)
           
            if  (labels < 0 ).any() or (labels >= num_bins).any():
                continue
            
            # Binned labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)
            
            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_cross_yaw = criterion_yaw(yaw, label_yaw)
            loss_cross_pitch = criterion_pitch(pitch, label_pitch)
            loss_cross_roll = criterion_roll(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)
            
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw = loss_cross_yaw + alpha * loss_reg_yaw
            loss_pitch = loss_cross_pitch + alpha * loss_reg_pitch
            loss_roll = loss_cross_roll + alpha * loss_reg_roll

            val_loss_lists = update_loss_history(val_loss_lists, loss_cross_yaw.item(), loss_cross_pitch.item(), loss_cross_roll.item(), \
                                loss_reg_yaw.item(), loss_reg_pitch.item(), loss_reg_roll.item(),\
                                loss_yaw.item(), loss_pitch.item(), loss_roll.item())

        val_loss_history_lists = update_loss_history(val_loss_history_lists, np.mean(val_loss_lists[0]), np.mean(val_loss_lists[1]),\
                            np.mean(val_loss_lists[2]), np.mean(val_loss_lists[3]), np.mean(val_loss_lists[4]), np.mean(val_loss_lists[5]),\
                            np.mean(val_loss_lists[6]), np.mean(val_loss_lists[7]), np.mean(val_loss_lists[8]))

        # Log and plot results 
        log_results(train_loss_history_lists[0], val_loss_history_lists[0], output_dir, 'cross_entropy_yaw', 1)
        log_results(train_loss_history_lists[1], val_loss_history_lists[1], output_dir, 'cross_entropy_pitch', 2)
        log_results(train_loss_history_lists[2], val_loss_history_lists[2], output_dir, 'cross_entropy_roll', 3)
        
        log_results(train_loss_history_lists[3], val_loss_history_lists[3], output_dir, 'MSE_yaw', 4)
        log_results(train_loss_history_lists[4], val_loss_history_lists[4], output_dir, 'MSE_pitch', 5)
        log_results(train_loss_history_lists[5], val_loss_history_lists[5], output_dir, 'MSE_roll', 6)
        
        log_results(train_loss_history_lists[6], val_loss_history_lists[6], output_dir, 'Total_yaw', 7)
        log_results(train_loss_history_lists[7], val_loss_history_lists[7], output_dir, 'Total_pitch', 8)
        log_results(train_loss_history_lists[8], val_loss_history_lists[8], output_dir, 'Total_roll', 9)
        
        print('Epoch [%d/%d],  Losses: Train Yaw %.4f, Val Yaw %.4f  || Train Pitch %.4f, Val Pitch %.4f ||  Train Roll %.4f, Val Roll %.4f  '
                       %(epoch+1, num_epochs, train_loss_history_lists[6][-1], val_loss_history_lists[6][-1], \
                        train_loss_history_lists[7][-1], val_loss_history_lists[7][-1],   
                        train_loss_history_lists[8][-1], val_loss_history_lists[8][-1]))

        if not os.path.exists(output_dir + '/output/'):
            os.mkdir(output_dir + '/output/')
        if not os.path.exists(output_dir + '/output/snapshots/'):
            os.mkdir(output_dir + '/output/snapshots/')
            
        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...')
            torch.save(model.state_dict(), output_dir +
            '/output/snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pkl')

        # saving best model
        total_val_loss = val_loss_history_lists[6][-1] + val_loss_history_lists[7][-1] + val_loss_history_lists[8][-1] 
        if total_val_loss < best_loss:
            best_loss = total_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir +
            '/output/snapshots/' + args.output_string + '_best_model' + '.pkl')


    ##########################################
    ##           Testing Model              ##
    ##########################################
    best_state_dict = torch.load(output_dir +'/output/snapshots/' + args.output_string + '_best_model' + '.pkl')
    #best_model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins)
    
    model.load_state_dict(best_state_dict)
    best_model = model
    best_model.cuda(gpu)

    best_model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    
    ## Testing best model on test set:
    test_filename_list = os.path.join(args.filename_list ,'test_filename.npy')
    test_dir = "/usr/stud/sharmaki/Projects/Advertima/datasets/AFLW2000-3D/AFLW2000/"
    pose_dataset_test = datasets.AFLW2000(test_dir, num_bins, test_filename_list, transformations)

    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset_test,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4)
    total = 0
    yaw_error = .0
    pitch_error = .0
    roll_error = .0
    print('Ready to test network on test data .......')

    l1loss = torch.nn.L1Loss(size_average=False)

    if not os.path.exists(os.path.join(output_dir,'Test_images' )):
        os.mkdir(os.path.join(output_dir, 'Test_images'))

    for i, (images, labels, cont_labels, name) in enumerate(test_loader):
        images = Variable(images).cuda(gpu)
        total += cont_labels.size(0)

        label_yaw = cont_labels[:,0].float()
        label_pitch = cont_labels[:,1].float()
        label_roll = cont_labels[:,2].float()

        yaw, pitch, roll = model(images)

        # Binned predictions
        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

        if True:
            name = name[0].split('.')[0]
            if args.dataset == 'BIWI':
                cv2_img = cv2.imread(os.path.join(args.data_dir, name + '_rgb.png'))
            else:
                cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg'))
            if args.batch_size == 1:
                error_string = 'y %.2f, p %.2f, r %.2f' % (torch.sum(torch.abs(yaw_predicted - label_yaw)), torch.sum(torch.abs(pitch_predicted - label_pitch)), torch.sum(torch.abs(roll_predicted - label_roll)))
                cv2.putText(cv2_img, error_string, (30, cv2_img.shape[0]- 30), fontFace=1, fontScale=1, color=(0,0,255), thickness=2)
            utils.draw_axis(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], tdx = 200, tdy= 200, size=100)
            cv2.imwrite(os.path.join(output_dir + '/Test_images/', name + '.jpg'), cv2_img)

        # Mean absolute error
        yaw_err = torch.sum(torch.abs(yaw_predicted - label_yaw))
        yaw_error += yaw_err
        pitch_err = torch.sum(torch.abs(pitch_predicted - label_pitch))
        pitch_error += pitch_err
        roll_err = torch.sum(torch.abs(roll_predicted - label_roll))
        roll_error += roll_err
        
        if (i+1) % 10 == 0:
            print('Iter [%d/%d] MAE Error: Yaw %.4f   ||   Pitch %.4f  ||   Roll %.4f'
                       %( i+1, len(pose_dataset_test)//args.batch_size, yaw_err.item(), pitch_err.item(), roll_err.item()))

    yaw_err = yaw_error / total
    pitch_err = pitch_error / total
    roll_err = roll_error / total
    
    total_err = (yaw_err + pitch_err + roll_err)/3
    print('Test error in degrees of the model on the ' + str(total) +
    ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, total_error: %.4f' %( yaw_err,
    pitch_err, roll_err, total_err))

    # Logging test results
    Error_file  = open(output_dir +'/output/'  + 'test_result_best_model.txt', "w")
    Error_file.write('Yaw Err: ' + str(yaw_err))
    Error_file.write('\n Pitch Err: '+  str(pitch_err))
    Error_file.write('\n Roll Err: '+  str(roll_err))
    Error_file.write('\n Avg Err: '+  str(total_err))
    Error_file.write('\n Best Model Epoch: '+  str(best_epoch))
    Error_file.close()

    print('Finished !!!')