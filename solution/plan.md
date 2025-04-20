Great, I’ll analyze how to design PLDM components using the Overcooked-AI dataset you provided. Since each row represents a timestep with joint actions, I’ll focus on a joint-agent model, which typically performs better in symmetric cooperative tasks like Overcooked.

I'll provide detailed design guidelines for the state encoder, dynamics predictor, and reward model based on the structure of the dataset.

# Designing a Joint-Agent Dynamics Model (PLDM) for Overcooked-AI

**Objective:** Build a predictive model (PLDM-like architecture) that learns the **environment dynamics and rewards** in Overcooked-AI from data. We use a **joint-agent modeling approach** (centralized model) since the task is fully cooperative and symmetric between the two agents. This model will consist of: 

- A **State Encoder** to represent the game state (both agents and all objects) in a neural-friendly format.  
- A **Dynamics Predictor** to predict the next state given the current state and *joint action* (actions of both agents).  
- A **Reward Predictor** to predict the immediate reward from the current state (and possibly action).  

We assume we have a dataset (e.g. CSV logs) where each row is a time step with a JSON state description, the joint action `[action_0, action_1]`, and the resulting reward. Below, we outline how to design each component, with practical tips on data encoding, network architecture (using PyTorch), and preparing input/output tensors.

## State Encoder

 ([Illustration of Overcooked environment. We choose two layouts Forced... | Download Scientific Diagram](https://www.researchgate.net/figure/Illustration-of-Overcooked-environment-We-choose-two-layouts-Forced-Coordination-and_fig1_382971150#:~:text=,)) ([Overcooked-AI game dynamics. | Download Scientific Diagram](https://www.researchgate.net/figure/Overcooked-AI-game-dynamics_fig2_364448758)) Overcooked is a **grid-world kitchen** where two agents (“chefs”) must cooperatively cook and serve soup. The state includes each agent’s position and orientation, any item they carry (e.g. ingredient or dish), objects in the environment (ingredients on counters, pots with soup, etc.), and static features like walls or dispensers ([Illustration of Overcooked environment. We choose two layouts Forced... | Download Scientific Diagram](https://www.researchgate.net/figure/Illustration-of-Overcooked-environment-We-choose-two-layouts-Forced-Coordination-and_fig1_382971150#:~:text=,)). We need to convert this rich state into an input tensor for our neural network.

There are two common approaches to encode the state: **Grid-based tensor encoding** or **flat feature vector encoding**. In either case, we must capture all relevant entities (both agents and objects). Key state elements to represent include:

- **Agent locations**: Each agent’s grid position (row, col). This can be encoded as coordinates or one-hot on a grid. If using a grid tensor, we may dedicate a channel for agent 1 and agent 2 presence ([Overcooked in Thousands of Kitchens: Training Top Performing Agents in Under a Minute](https://bsarkar321.github.io/blog/overcooked_madrona/index.html#:~:text=Each%20agent%20is%20given%20an,this%20location%2C%20while%20feature%2021)). If using a feature vector, we can include the numeric positions (possibly normalized or one-hot encoded among possible tiles).  
- **Agent orientation**: The direction each agent is facing (e.g. up, down, left, right). We can encode this as a small one-hot vector of length 4 (or 5 if including a “no orientation” or idle) appended to the agent’s features.  
- **Held object**: What each agent is holding (if anything). For example, one-hot encode the item type (`none`, `onion`, `tomato`, `dish`, `soup`) or use an embedding vector (e.g. map each item type to an integer and embed in, say, 4-8 dimensions).  
- **Objects in the environment**: Represent the location and type of each relevant object not held by an agent. In Overcooked, objects like onions, tomatoes, dishes, or soup bowls may be on counters or in pots. A convenient method is to use a **grid of feature channels**: for example, a binary map for onions, another for tomatoes, another for dishes, etc., with a 1 indicating the presence of that object on each cell ([Overcooked in Thousands of Kitchens: Training Top Performing Agents in Under a Minute](https://bsarkar321.github.io/blog/overcooked_madrona/index.html#:~:text=Each%20agent%20is%20given%20an,this%20location%2C%20while%20feature%2021)). Pots could be represented by multiple features (e.g. one channel for an empty pot, one for a pot cooking, one for a pot with ready soup). If using a flat vector, alternatively maintain a list of object positions and types (e.g. you could have a fixed-size vector for pots including their status, and separate vectors for counts of onions on counters, etc., but a grid encoding is simpler to automate).  
- **Static layout**: The fixed features of the map such as walls, floor, dispensers (onion supply, dish supply), serving window, etc. If you use a CNN-based grid encoding, you can include constant channels that mark these locations (e.g. a wall channel, a dispenser channel) so the network knows the layout. If using a flat vector, you might not need to encode static layout if you assume the network is specialized per layout, but for generality it’s better to include it (e.g. one-hot encode the layout name, or include binary flags for presence of certain structures at the agents’ neighboring cells). In many cases, the layout is known and can be baked into the model separately (for example, by using the same input channels for walls/counters every time).  

**Grid Encoding Approach:** Represent the state as a 3D tensor of shape `(height, width, channels)` encoding all information. For example, Overcooked uses a 26-channel binary encoding for each cell in a 5x7 grid in one implementation ([Overcooked in Thousands of Kitchens: Training Top Performing Agents in Under a Minute](https://bsarkar321.github.io/blog/overcooked_madrona/index.html#:~:text=Each%20agent%20is%20given%20an,this%20location%2C%20while%20feature%2021)). In our case, channels could include: one channel per agent (agent’s presence), one per object type, one per goal or dispenser, etc. This tensor can be fed into a convolutional neural network. The advantage is that spatial relationships are preserved (useful since the outcome of actions depends on neighbors, e.g. picking up an item requires an agent adjacent to it). If we choose this approach, our **State Encoder** can simply be a stack of convolutional layers that process this multi-channel image and output a state embedding vector. (For example, 2-3 conv layers with ReLU activation and maybe a flatten at the end to produce a 128-dim vector.)

**Feature Vector Approach:** Alternatively, encode the state as a concatenated feature vector. For each agent and object, add their features to a single 1D tensor. For instance, we might create a vector like: 

```
[ agent1_x, agent1_y, agent1_orient_onehot(4 dims), agent1_heldItem_onehot(N dims),
  agent2_x, agent2_y, agent2_orient_onehot(4 dims), agent2_heldItem_onehot(N dims),
  object1_type_onehot *and* object1_position, object2_type, object2_position, ...,
  time_left (optional), etc. ]
```

This would yield a fixed-length vector representation of the state. We may need to decide on a maximum number of objects to encode explicitly (e.g. at most one soup, etc.) or otherwise aggregate objects (e.g. count of onions in pot). Simpler: since the dataset’s JSON has an `"objects"` list, we could include features for each pot’s status and maybe ignore loose ingredients for the dynamics model if they’re always picked up or in pot (for completeness, you could include up to M loose items with their positions). In practice, because Overcooked is fairly structured (at most a few items on the ground or in pots), a fixed-size vector is feasible.

**Symmetry and Joint Encoding:** Because the game is symmetric w.r.t. the two agents (they have identical roles), our encoding and model should treat them equitably. In practice, that means we **include both agents’ info in the state representation**, rather than modeling one agent at a time. We can simply concatenate [agent1 features, agent2 features,…] in a fixed order (player 0 vs player 1 as given). To mitigate any arbitrary ordering, one trick is to also train with data where we swap agent identities (and correspondingly swap their actions) as augmentation, so the model doesn’t get biased to a particular ordering. Another approach is to process each agent’s features through the *same* sub-network (sharing weights) and then aggregate (e.g. average or sum) to form a joint representation, making the encoding **permutation-invariant** to agent order. For simplicity, one can start without explicit permutation invariance (the dataset likely has a consistent ordering), but keep in mind the symmetry for designing the model. 

**Output of State Encoder:** The end result of the state encoder is an **embedding vector** (let’s say `state_emb` of size 128, as a design choice). If using a CNN, this is the flattened output after conv layers. If using a flat vector input, the state encoder might just be an identity (pass the features through) or a few fully connected layers to transform it into a more suitable latent space. We could, for example, use a fully-connected layer that maps the concatenated input features to a 128-dim latent vector (with ReLU activation), serving as the encoded state. This latent state representation will then be fed into the Dynamics and Reward predictor networks.

## Dynamics Predictor (Next-State Model)

The **Dynamics Predictor** is a neural network `f` that takes as input the **current state** (encoded) and the **joint action** of both agents, and outputs a prediction of the **next state**. Formally, we want `f(s_t, a_t) ≈ s_{t+1}`. In our case, the joint action $a_t$ consists of agent0’s action and agent1’s action together. We will encode the joint action and feed it alongside the state embedding. This centralized dynamics model naturally considers the *collective action of both agents* when predicting the next state ([DeepSafeMPC: Deep Learning-Based Model Predictive Control for Safe Multi-Agent Reinforcement Learning](https://arxiv.org/html/2403.06397v1#:~:text=multi,has%20shown%20promising%20results%20in)).

**Action Encoding:** Each agent’s action in Overcooked can be one of several discrete choices (move in four directions, do nothing, or interact/pickup). From the dataset, we see actions represented as `[dx, dy]` for movement or strings like `"interact"` for pickup/drop. We should convert each agent’s action into a uniform encoding. A convenient scheme is to define an **action index** for each possible action: e.g. 0=NOP (no-op), 1=Up, 2=Down, 3=Left, 4=Right, 5=Interact. Each agent’s action can then be represented as a one-hot vector of length 6 (if we include the “stay” action) or length 5 if we consider [0,0] as a form of no-op already included. In the dataset snippet, `[0,1]` likely corresponds to “move right,” `[0,0]` is “do nothing,” and `"interact"` is the pickup/drop action. We’ll map these to our indices accordingly. 

We then create a **joint-action representation** by either concatenating the two one-hot vectors (resulting in a length 12 vector if each is length 6), or by passing each through an embedding layer and concatenating the embeddings. For example, we could have an `nn.Embedding(6, 4)` in PyTorch to turn each agent’s action into a 4-dim trainable embedding, and then concatenate those to get an 8-dim joint action embedding. This embedding (or concatenated one-hots) will be fed into the dynamics model along with the state embedding. 

**Network Architecture:** We can implement the dynamics predictor as a **fully connected neural network** (multilayer perceptron). Its input will be the concatenation of the state embedding and action encoding. For instance, if `state_emb` is 128-d and `action_emb` is 8-d, the input to this network is a 136-dimensional vector. We then pass it through a few hidden layers to produce an output that represents the next state. A possible architecture in PyTorch:

- Input layer: size 136 → Hidden layer 1: e.g. 128 neurons, ReLU activation.  
- Hidden layer 2: e.g. 128 neurons, ReLU. (We can add 2-3 layers of 128 or 256 units depending on complexity needed. Overcooked dynamics are fairly straightforward, so 2 layers of 128 might suffice, but more complex interactions might benefit from bigger layers like 256.)  
- Output layer: produces a vector representing the next state. The output size should match our state encoding scheme. For example, if we encode state as a vector of length $D$, the model could output a $D$-dim vector $\hat{s}_{t+1}`. If we use a CNN-based state encoding, another design is to have the model output parameters that can be decoded into a grid – but it’s simpler to output the same kind of encoded features as the state encoder produces (e.g. predict positions, etc., in a flat vector).  

**Predicting a complex state:** It’s worth noting that the next state prediction may involve discrete changes (like an agent’s position changing, an object disappearing from one tile and appearing in another). You have two options for formulating the output: 
- **Direct regression:** Treat the next state features as continuous or one-hot targets and use regression (MSE loss) to predict them. For example, you could have the network directly output the next coordinates of each agent (as numbers) and indicators if they carry an object, etc. This might require something like a custom loss for classification (if you output a probability distribution for next position on the grid).  
- **Structured output:** Alternatively, break the output prediction into parts. For example, you might have the network output two 2D vectors for the agents’ next positions (or a probability map for each agent’s position on the grid of size width×height), plus additional outputs for whether an object was picked up or delivered. This can get complicated. 

A simpler approach is to stick with a **single-vector output** equal to the encoded next state. Since our state encoding includes things like agent positions (which are discrete), we may interpret the network’s output in training with a suitable loss. For instance, if we encode agent position as a one-hot over grid cells, we can use a softmax + cross-entropy loss for that part. If we encode it as two numbers (x,y), we can use MSE loss. Similarly, for categorical aspects (like held object type), use cross-entropy; for binary flags, use binary cross-entropy or MSE. In practice, a combination of losses or a custom loss function might be used to train the model to match all parts of the next-state vector. But to keep things straightforward, we might encode everything as numbers (0/1 or 0/… for categories) and just use MSE on the whole vector — it will treat one-hot vectors as multi-output regression which often works if properly normalized. 

**Example:** If agent positions are one-hot on a 5×7 grid (35 dims each), the network’s output for those can be interpreted as predicted probabilities for each cell; we could apply a softmax over those 35 outputs and choose the argmax as predicted position. Training can minimize cross-entropy with the true cell index. However, if implementing from scratch, one could avoid the softmax and just use MSE on the one-hot (which will encourage the correct cell output to be 1). Both approaches are viable. The key is to ensure the model’s output dimension and the target representation align.

**Implementation Tips:** In PyTorch, you would define a module `DynamicsModel(nn.Module)` with something like: 

```python
self.state_encoder = ...  # e.g., an MLP or CNN to embed state
self.action_emb = nn.Embedding(num_actions, action_emb_dim)  # for each agent
self.fc1 = nn.Linear(state_emb_dim + 2*action_emb_dim, 128)
self.fc2 = nn.Linear(128, 128)
self.fc_out = nn.Linear(128, state_output_dim)
```

During `forward`, you would encode the state (or if state is already numeric vector, maybe just pass it), embed each agent’s action, concatenate all, then apply the layers. Use ReLU between fc layers. For training, you’ll compare `pred_next_state = model(state, action)` with the true next state vector from the dataset. 

This **joint dynamic model** inherently captures interactions: since both agents’ actions are input simultaneously, the network can learn effects like two agents bumping into each other or one picking up an item that another was going for. Modeling jointly is crucial in a cooperative task because the next state is determined by the combination of both agents’ moves ([DeepSafeMPC: Deep Learning-Based Model Predictive Control for Safe Multi-Agent Reinforcement Learning](https://arxiv.org/html/2403.06397v1#:~:text=multi,has%20shown%20promising%20results%20in)). (For example, if agent1 and agent2 both reach the same soup pot with an onion and a dish respectively, the next state could have the onion in the pot and the dish still with agent2, etc.) A single-agent transition model would fail to capture such coordinated effects.

**Symmetric considerations:** As mentioned, if we wanted, we could share the state encoder for each agent. But since we already encoded the whole state as one vector, the network will just learn the interactions. If we do use a CNN on the grid, that CNN naturally treats both agents uniformly if they are encoded in separate channels (one channel for agent1, one for agent2). The convolutional filters will pick up either agent similarly. That provides some inherent symmetry. If using a flat vector and we are concerned about ordering bias, one could enforce symmetry by, say, always sorting the two agents by some criteria (like their ID or positions) – but in practice, this may not be necessary if we train on both roles sufficiently. 

## Reward Predictor

The **Reward Predictor** is a model $r(s_t, a_t) \approx R_{t}$ that estimates the immediate reward given the current state (and possibly the action). In Overcooked, the reward is typically sparse – e.g., delivering a soup gives +20 reward, otherwise 0 most time steps (some versions might also have a time penalty). The dataset’s `reward` column for each timestep is the reward achieved after that joint action. We want to train a network to predict this from the state (and action), analogous to a learned reward function in model-based RL ([](https://amslaurea.unibo.it/id/eprint/27600/1/Ferraioli_Valentina_tesi.pdf#:~:text=instance%2C%20the%20model%20can%20predict,interact%20in%20the%20following%20way)).

**Input:** The input to the reward model can be the **current state encoding** and optionally the joint action. Why include action? Because certain rewards only happen on specific actions (e.g. the act of delivering a soup yields the reward). If the state alone can tell (for instance, maybe seeing a soup at the serving station could imply a delivery), but it’s cleaner to also give the action as input. For example, if one agent’s action is “interact” at the serving counter while holding a soup, that will produce a reward. The model can learn this easier if it knows the action was an interact. However, you might also design the state to include enough info (like an indicator that a delivery happened), but that blurs the line between state and action. So, we will feed both. 

Thus, we use the same state embedding (`state_emb`) from the State Encoder, and the same joint action encoding as before. We concatenate them (just like in the dynamics model). This combined vector goes into the reward network.

**Architecture:** The reward predictor can be a **small MLP** since it outputs just a single value. For example:

- Input: `state_emb_dim + action_emb_dim_total` (e.g. 128+8 = 136 as before).  
- Hidden layer: 64 neurons, ReLU (reward function is usually simpler than dynamics, so a smaller network can suffice).  
- Maybe a second hidden layer: 64 neurons, ReLU (or you can even use one layer if empirical results are fine).  
- Output layer: 1 neuron, linear output for the predicted reward. 

Since Overcooked rewards are numeric (0 or 20 typically), we can treat this as a regression problem. We’ll train the network with a regression loss (mean-squared error, or even better mean absolute error if we expect mostly 0s and some 20s – MAE might be more robust for sparse rewards). If we know rewards are always integer, we could also formulate it as classification (chance of getting 20 vs 0), but regression is straightforward.

**Sharing vs Separate Encoder:** One practical tip is **weight sharing** – we could reuse the state encoder’s output from the dynamics model for reward prediction to avoid redundant computation. For instance, if you already computed a 128-dim `state_emb` for the dynamics model, you can feed that into the reward MLP head. In implementation, you might make the dynamics model and reward model separate modules, but have them use a common sub-module for the state encoder. This way, the same state features are used for both predictions (this can be jointly trained if desired, or sequentially). However, be careful if training jointly: the dynamics loss and reward loss have different scales; you may want to balance them. If it’s simpler, you can train the dynamics and reward models separately (first train the dynamics model network, then freeze or copy the encoder and train a reward model on top). 

If not sharing, just know you’ll have two networks processing the state – which is fine if not too expensive (Overcooked state isn’t huge, so this is acceptable as well). 

**Using State or (State, Next State):** In some cases, a reward depends on *state transition* rather than state alone (e.g. a delivery might be detected by soup leaving inventory and appearing at serving in next state). But since we include the action, the model can infer reward without explicitly seeing the next state. For example, if `action_agent1 == interact` and agent1 is at serving counter with soup in hand (info available in current state) then reward = 20. Our model can learn this rule. So including action makes it Markov. If one didn’t include action, the model would have to guess if an interact happened – not ideal. Thus, we feed $(s_t, a_t)$ to predict $r_t$.

**Training:** We prepare training pairs where input is state+action, target is the `reward` from that timestep in the dataset. We minimize MSE loss between predicted and actual reward. Because reward in Overcooked is often zero, the model may initially just predict 0 everywhere – one may need to ensure it gets enough examples of positive reward to learn those conditions (the dataset should include those when a soup is delivered). In practice, the model will likely learn a near-zero output with occasional spikes for those specific conditions.

**Architecture Example (PyTorch):** 

```python
class RewardModel(nn.Module):
    def __init__(self, state_emb_dim, action_emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_emb_dim + action_emb_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 1)
    def forward(self, state_emb, action_emb):
        x = torch.cat([state_emb, action_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        r = self.fc_out(x)  # no activation (regression output)
        return r
```

Where `state_emb` could either be the raw encoded state vector or output of a shared encoder; `action_emb` is the concatenated embedding for joint action (as used in dynamics model). This will output a single scalar per sample. 

In summary, the reward predictor is a simple model that captures the **reward function** of the environment. In model-based RL terms, along with the dynamics predictor, it forms the learned model of the MDP ([](https://amslaurea.unibo.it/id/eprint/27600/1/Ferraioli_Valentina_tesi.pdf#:~:text=instance%2C%20the%20model%20can%20predict,interact%20in%20the%20following%20way)) (transition function and reward function). 

## Preparing Input/Output Tensors from the Dataset

Finally, we address how to go from the provided Overcooked dataset (CSV of timesteps) to training data for our models. Each row in the dataset contains a JSON `state`, a `joint_action`, and the resulting `reward`, among other info (like time left, score, etc.). To train the dynamics model, we need $(s_t, a_t)$ as input and $s_{t+1}$ as target. To train the reward model, we need $(s_t, a_t)$ as input and $r_t$ as target. We outline a procedure:

1. **Parse the CSV**: Load the dataset using Python (e.g. the `csv` module or pandas). Each row is one timestep. Important: Identify which rows belong to the same game episode (`trial_id` or similar column can group these). Sort by time or timestep within each episode. 

2. **Extract State and Action**: For each row, parse the `state` JSON string. In Python, you can do `import json; state_dict = json.loads(row['state'])`. This gives a dictionary with keys like `"players"` (list of two players) and `"objects"` (list of objects in the world). From this, extract the features needed:
   - Player positions: `state_dict["players"][0]["position"]` might give something like `[8,3]`. Orientation: `state_dict["players"][0]["orientation"]` maybe `[0,-1]` (a vector). You’ll need to convert orientation vector to a direction index or one-hot (e.g. `[0,-1]` might mean “up” if we interpret that as Δrow=-1). Held object: e.g. `state_dict["players"][0]["held_object"]` could be `null` or an object dict (with a `"name"` like "dish"). Convert that to our item encoding (none/dish/etc). Do the same for the second player.
   - Objects: iterate `state_dict["objects"]` list. Each object might have a `"name"` ("onion", "soup", etc.) and a `"position"`. Use this to mark the presence on your grid or to fill in your object features. For example, if an object is a soup, you might note the soup’s position and whether it’s cooking or ready (there are fields like `"is_ready"` in the soup object). These details might be optional for your model; including them makes it more accurate. At least, one should encode that a soup exists at that location (which implies perhaps an agent could deliver it next).
   - Orders/bonus orders: Overcooked state also lists current orders (`"all_orders"`). For dynamics and low-level state transitions, you usually don’t need to encode the orders into the state, since they don’t directly affect physics – they affect long-term reward but not immediate transitions. However, if you wanted a more comprehensive state (for planning full game), you could include a feature for the type of soup currently required or being made. For simplicity, we can ignore orders in the state encoding for the dynamics model, because whether an order is onion or tomato doesn’t change the movement dynamics. It only matters for reward (delivering the correct soup yields reward). In Overcooked, any soup delivered yields reward, so even then the specific order might not matter for immediate reward prediction (unless partial credit for wrong soup, which standard Overcooked doesn’t do). So we can skip orders in our encoding.
   - Time left or timestep: The dataset has a `time_left` and `timestep`. These likely don’t affect dynamics except that when time runs out the episode ends. You might include `time_left` as an input feature if you want the model to predict end-of-episode dynamics or something, but it’s not necessary if we assume episodes are truncated when time_left=0. It could help the model know that after time runs out, no next state (or terminal state). For simplicity, you can ignore time in the state encoding, or include it as just a number if you want the model to potentially learn that when time=0, maybe everything resets (but your training data likely won’t include a transition after the last step anyway).
   - Static layout: If you want to use a CNN with the static layout, you might load the `layout` field from the dataset (it shows a grid of X, O, etc. in the CSV). You can parse that or use a predefined layout encoding. Alternatively, since each trial might have a known layout name, you can load a predefined encoding for that layout. In any case, if including static features, add those channels (like walls) in the state representation consistently for all time steps of that layout.

3. **Encode State**: Using the chosen representation (grid or vector), convert the parsed state into a tensor:
   - For grid encoding: initialize an array of zeros with shape `(channels, height, width)`. Mark agent1’s position in agent1-channel, agent2’s in agent2-channel (e.g. set [agent1_channel, y, x] = 1). Mark objects: e.g. for each onion object, set [onion_channel, y, x] = 1, etc. Static: mark walls (e.g. set [wall_channel, y, x] = 1 for all wall positions). This yields a tensor per state. You might then pass this through your conv net encoder to get `state_emb`. If you plan to train end-to-end, you’d feed this to the network directly. If you plan to pre-encode states with a separate encoder step, you could flatten it. In PyTorch, it’s often easiest to include the conv as part of the model and feed raw grids.
   - For vector encoding: create a vector (NumPy array) for the state features as designed. Append each feature in a consistent order. E.g. `[p0_x, p0_y, p0_orient_idx, p0_hold_idx, p1_x, p1_y, p1_orient_idx, p1_hold_idx, pot1_status, pot1_contents, pot1_position, ...]`. Ensure you use the same order for every time step. If some object is absent (e.g. a second pot might not exist in a layout), you could use zeros or a special “none” encoding in that slot. The vector length must remain fixed. After assembling, convert this to a PyTorch tensor (float dtype). You might also normalize certain numeric values (for example, divide coordinates by the grid width/height to scale 0-1, though for small integers it may not be necessary).

4. **Encode Action**: Parse the `joint_action` field. In the CSV it appears as a string like `"[[0, 1], \"interact\"]"` (one agent moved, the other interacted). You can use `ast.literal_eval` or `json.loads` on a slightly massaged string to turn this into a Python list. Suppose you get `action = [[0,1], "interact"]` for two agents. Map each to your action index: e.g. `[0,1]` → Right (assuming that’s how we define it), `"interact"` → Interact. So you might get indices [4, 5] (if 4=Right, 5=Interact). Then turn that into either two one-hot vectors or two embedding IDs. If you are using an embedding layer in the model, you will just store the two integers (for later feeding into the model’s embedding). If not, create a one-hot of length 6 for each and concatenate (resulting in length 12 vector). This forms the **action input**. In code, if using one-hots, you can do `action_vec = torch.zeros( (batch_size, 12) ); action_vec[i, idx_agent0] = 1; action_vec[i, 6+idx_agent1] = 1`. If using embedding, you might instead keep `action_indices = (idx_agent0, idx_agent1)` and feed them to the model which does the embedding internally. 

5. **Prepare Next State (for dynamics model)**: We need the next state as the target for the dynamics predictor. Given we have each row’s current state, the next state corresponds to the *next row* in the same episode. So, as you iterate through time steps, for each time step $t$ (except the last of an episode), you will pair it with the state at $t+1$. In practice:
   - Group data by `trial_id`. For each group (episode), sort by `timestep` (or time). 
   - For each consecutive pair in that sorted order, let the earlier be current state and later be next state.
   - Store the encoded current state, encoded action, and encoded next state. Also store the immediate reward (which should match the current state’s reward field).
   - You should be careful not to create a pair when a new trial starts (i.e., the last step of an episode has no next state in the same episode). Typically, you will just naturally handle that by iterating length-1 times for each episode.
   - If an episode ends early (e.g. time run out), the last state's action might not have a meaningful “next” (if environment terminates). But usually, you’ll still have that last step logged, possibly with a reward (e.g. if a delivery happened at final second). You can choose to exclude the final step from dynamics training (since no next state) or include it if the environment logs a final state after action. Check if the data includes a terminal next state; likely not explicitly. Safer to skip transitions where `timestep` resets or `time_left` increases (sign of new episode).

6. **Form Tensors and Datasets:** Once you have lots of $(s_{\text{enc}}, a_{\text{enc}}, s'_{\text{enc}}, r)$ samples, create PyTorch tensors/datasets:
   - `state_tensor`: shape `(N, state_dim)` or `(N, channels, H, W)` for N samples.
   - `action_tensor`: maybe split into two `(N,)` arrays for agent actions if using embedding, or a single `(N, action_dim)` if using one-hot concatenation.
   - `next_state_tensor`: shape `(N, state_dim)` (target for dynamics).
   - `reward_tensor`: shape `(N, 1)` or `(N,)` (target for reward, float). 

   You can use `torch.utils.data.TensorDataset` to combine these and then a DataLoader for batching. 

7. **Training Loop:** For the dynamics model, use a loss that suits your output. A simple approach is Mean Squared Error: `loss = MSE(pred_next_state, true_next_state)`. This works if your state encoding is numeric. If you included many one-hots in the encoding, MSE treats matching 1s and 0s similarly to cross-entropy in a simplistic way. You could improve by splitting the loss: e.g. use `CrossEntropyLoss` for agent position (if you output raw logits for positions rather than one-hots), and MSE for continuous stuff. But to keep it simple, MSE on the whole vector can do an acceptable job if scaled properly. Ensure you mask or handle any parts of the state that are not used; ideally your encoding includes everything needed. Train for enough epochs until loss plateaus. Monitor qualitatively if possible (e.g. check if the model correctly predicts position changes). 
   
   For the reward model, use `MSELoss(pred_reward, true_reward)`. Since reward is often 0, the model might predict something slightly off (like 0.1) if not careful – sometimes clipping or using Huber loss could be useful if outliers matter. But given the simplicity (predict 0 or 20), it will likely converge to predicting near 0 for no reward and ~20 for those states that lead to reward. You could also treat it as classification (reward >0 or not) plus magnitude, but not necessary.

8. **Architectural Tuning:** Some tips – the sizes (128 for state embedding, etc.) are not fixed rules. If the state encoding vector is very large, you might increase layer sizes. If using CNN, ensure the conv layers compress the grid (maybe use a couple conv + a flatten + an FC). If the model has trouble, adding a bit more capacity (more neurons or an extra layer) might help. Conversely, if overfitting (loss on training much lower than validation), you might reduce capacity or add regularization (dropout layers in the MLP or L2 weight decay). 

9. **Symmetric Data Augmentation:** Since agents are interchangeable, you can augment the training data by swapping agent 0 and 1 in the state and action, and using the same next state swapped. This doubles data and helps the model not assign meaning to “which agent is which”. This is optional and can be done by programmatically swapping features for half the dataset.

10. **Testing the model:** After training, you can feed a state and joint action from the dataset into the dynamics predictor and see if the predicted next state matches the actual next state. Similarly, test the reward predictor on some examples (it should output ~20 when an agent is delivering soup, otherwise ~0). 

By following these steps, we prepare our data and design our PLDM-like model. In summary, we represent the Overcooked state with appropriate encodings (ensuring both agents and all objects are included), use a joint action encoding, and train a neural network to learn $F(s,a) \to s'$ and $G(s,a) \to r$. This model will then serve as a learned simulator of the Overcooked environment’s dynamics and reward structure ([](https://amslaurea.unibo.it/id/eprint/27600/1/Ferraioli_Valentina_tesi.pdf#:~:text=instance%2C%20the%20model%20can%20predict,interact%20in%20the%20following%20way)), which is particularly useful for planning or training new policies in a model-based reinforcement learning setting. The **joint-agent modeling approach** ensures the model captures the cooperative interactions in this symmetric environment, rather than treating agents in isolation. By using straightforward architectures (fully-connected layers for dynamics and reward, with an optional convolutional state encoder), we can implement this in PyTorch and train it on the provided dataset to accurately mimic the Overcooked-AI environment’s behavior. 


# Planning Module Implementation Plan for PLDM in Overcooked-AI

## Objective
To build an effective **planning component** within a PLDM-like (Planning with Learned Dynamics Model) architecture to select optimal joint-actions for cooperative agents in the Overcooked-AI environment.

## Approach

We propose using a **Model-Predictive Control (MPC)** approach, which involves:

- **Sampling-based Planning**: Use the learned dynamics and reward predictors to simulate multiple future trajectories.
- **Action Selection**: Evaluate sampled trajectories to select actions maximizing expected cumulative rewards.

## Implementation Steps

### 1. Trajectory Sampling
- Generate a batch of possible future joint-action sequences.
- Use stochastic action sampling (e.g., uniform sampling initially, later guided by learned policies).

### 2. Forward Simulation
- Simulate each trajectory using the trained **Dynamics Predictor** from the PLDM.
- Predict rewards for each state-action pair in these simulated rollouts using the **Reward Predictor**.

### 3. Evaluation and Selection
- Calculate cumulative predicted rewards for each trajectory.
- Select the initial joint-action from the trajectory with the highest cumulative reward as the optimal action to execute.

### 4. Optimization with Cross-Entropy Method (Optional)
- Initially, use uniform sampling. After baseline implementation, consider incorporating the Cross-Entropy Method (CEM) to iteratively improve action sampling distributions:
  - Sample trajectories from a distribution over actions.
  - Keep top trajectories (elite actions).
  - Fit and update a new action distribution based on elite actions.

## Challenges
- **Sparse Reward Issue**: Difficulty in differentiating trajectories due to limited immediate feedback.
  - **Mitigation**: Employ shaped rewards or intrinsic curiosity signals during planning.
- **Model Prediction Errors**: Accumulating errors in multi-step predictions.
  - **Mitigation**: Keep planning horizons moderate initially (short to medium length), progressively increasing as the model improves.

## Tools
- **PyTorch**: Implementing model predictions.
- **NumPy**: Trajectory sampling and reward calculation.
- **Parallelization**: Employ batch processing and GPU acceleration for efficient planning simulations.

## Initial Parameters
- **Planning Horizon**: Start with 5-10 steps initially.
- **Trajectory Batch Size**: Begin with 50-100 sampled trajectories per planning step.
- **Action Space**: Discrete joint-action sets for both agents.

## Validation
- Initially validate planning via manual inspection and baseline comparisons against random or heuristic-based action selections.
- Progressively evaluate performance improvement through iterative enhancements and CEM integration.