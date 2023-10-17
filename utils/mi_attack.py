import os
import math
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from copy import deepcopy
    
def inversion(agent, G, T, alpha, z_dim = 100, max_episodes=40000, max_step=1, label=0, model_name="VGG16"):
    print("Target Label : " + str(label))
    best_score = 0
    c = None

    for i_episode in range(1, max_episodes + 1):
        y = torch.tensor([label]).cuda()

        # Initialize the state at the beginning of each episode.
        z = torch.randn(1, G.z_dim).cuda()
        state = deepcopy(z.cpu().numpy())
        for t in range(max_step):
            # print(state.shape)
            # Update the state and create images from the updated state and action.
            action = agent.act(state)
            z = alpha * z + (1 - alpha) * action.clone().detach().reshape((1, len(action))).cuda()
            next_state = deepcopy(z.cpu().numpy())
            state_image = G(z, c).detach()
            state_image = F.interpolate(state_image, size=(64, 64), mode='bilinear', align_corners=False)
            # print(state_image.shape)
            action_image = G(action.clone().detach().reshape((1, len(action))).cuda(), c).detach()
            action_image = F.interpolate(action_image, size=(64, 64), mode='bilinear', align_corners=False)
            # Calculate the reward.
            _, state_output = T(state_image)
            _, action_output = T(action_image)
            score1 = float(torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(state_output, dim=-1)).data, 1, y))))
            score2 = float(torch.mean(torch.diag(torch.index_select(torch.log(F.softmax(action_output, dim=-1)).data, 1, y))))
            score3 = math.log(max(1e-7, float(torch.index_select(F.softmax(state_output, dim=-1).data, 1, y)) - float(torch.max(torch.cat((F.softmax(state_output, dim=-1)[0,:y],F.softmax(state_output, dim=-1)[0,y+1:])), dim=-1)[0])))
            reward = 2 * score1 + 2 * score2 + 4 * score3

            # Update policy.
            if t == max_step - 1 :
                done = True
            else :
                done = False
            
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
        
        # Save the image with the maximum confidence score.
        test_images = []
        test_scores = []
        for i in range(1):
            with torch.no_grad():
                z_test = torch.randn(1, G.z_dim).cuda()
                for t in range(max_step):
                    state_test = z_test.cpu().numpy()
                    action_test = agent.act(state_test)
                    z_test = alpha * z_test + (1 - alpha) * action_test.clone().detach().reshape((1, len(action_test))).cuda()
                test_image = G(z_test, c).detach()
                test_image = F.interpolate(test_image, size=(64, 64), mode='bilinear', align_corners=False)
                test_images.append(test_image.cpu())
                _, test_output = T(test_image)
                test_score = float(torch.mean(torch.diag(torch.index_select(F.softmax(test_output, dim=-1).data, 1, y))))
            test_scores.append(test_score)
        mean_score = sum(test_scores) / len(test_scores)
        if mean_score >= best_score:
            best_score = mean_score
            best_images = torch.vstack(test_images)
            os.makedirs("./result/images/{}".format(model_name), exist_ok=True)
            os.makedirs("./result/models/{}".format(model_name), exist_ok=True)
            save_image(best_images, "./result/images/{}/{}.png".format(model_name, label), nrow=10)
            torch.save(agent.actor_local.state_dict(), "./result/models/{}/actor_{}_{}.pt".format(model_name, label, alpha))
        if i_episode % 10000 == 0 or i_episode == max_episodes:
            print('Episodes {}/{}, Confidence score for the target model : {:.4f}'.format(i_episode, max_episodes,best_score))
            save_image(best_images, "./result/images/{}/final_image_{}_{}.png".format(model_name, i_episode, label), nrow=10)

    return best_images