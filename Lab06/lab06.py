import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Input sentence
sentence = ["I", "love", "deep", "learning"]

# Step 2: Raw attention scores (can be any values)
attention_scores = torch.tensor([0.1, 0.3, 0.4, 0.2])

# Step 3: Apply softmax to convert into probabilities
attention_weights = F.softmax(attention_scores, dim=0)

# Convert to numpy for plotting
attention_weights = attention_weights.detach().numpy()

# Step 4: Plot heatmap
plt.figure(figsize=(8, 2))
sns.heatmap(
    [attention_weights],
    annot=True,
    cmap="Blues",
    xticklabels=sentence,
    yticklabels=["Attention"],
    cbar=True
)

plt.title("Attention Heatmap Visualization")
plt.xlabel("Words")
plt.ylabel("")
plt.show()