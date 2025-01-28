// URL du modèle ONNX
const MODEL_URL = "./static/model/fashion_mnist.onnx";

// Classes des vêtements
const classes = [
  "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
];

// Variable pour suivre l'état du modèle chargé
let session;
let isModelLoaded = false;

// Charger le modèle ONNX
(async () => {
  session = await ort.InferenceSession.create(MODEL_URL);
  isModelLoaded = true;
  console.log("Modèle chargé !");
})();

// Prétraitement de l'image (Transformation)
function transformImage(img) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  // Redimensionner à 28x28
  canvas.width = 28;
  canvas.height = 28;

  // Dessiner l'image redimensionnée
  ctx.drawImage(img, 0, 0, 28, 28);

  // Obtenir les données de l'image
  const imageData = ctx.getImageData(0, 0, 28, 28);
  const data = imageData.data;

  // Transformer chaque pixel
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];

    // Convertir en niveaux de gris
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;

    // Si le pixel est proche du blanc (considéré comme fond), rendre noir
    if (gray > 200) {
      data[i] = 0;     // R
      data[i + 1] = 0; // G
      data[i + 2] = 0; // B
    } else {
      // Sinon, garder les nuances de gris
      data[i] = gray;
      data[i + 1] = gray;
      data[i + 2] = gray;
    }
  }

  // Mettre à jour l'image transformée
  ctx.putImageData(imageData, 0, 0);

  return canvas;
}

// Faire une prédiction
async function predict(transformedCanvas) {
  // Obtenir les données de l'image transformée
  const ctx = transformedCanvas.getContext("2d");
  const imageData = ctx.getImageData(0, 0, 28, 28).data;

  // Préparer les données pour le modèle
  const input = [];
  for (let i = 0; i < imageData.length; i += 4) {
    const gray = imageData[i] / 255.0; // Normaliser entre 0 et 1
    input.push((gray - 0.5) / 0.5);   // Normalisation -1 à 1
  }

  // Créer un tenseur pour le modèle ONNX
  const tensor = new ort.Tensor("float32", new Float32Array(input), [1, 1, 28, 28]);

  // Faire une prédiction avec ONNX Runtime
  const results = await session.run({ image: tensor });
  const logits = results.preds.data; // Logits bruts du modèle

  // Identifier la classe prédite
  const predictedIndex = logits.indexOf(Math.max(...logits));

  return classes[predictedIndex];
}

// Gestion de la sélection d'image
document.getElementById("imageInput").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) {
    alert("Veuillez sélectionner une image.");
    return;
  }

  // Charger l'image
  const img = await loadImage(file);

  // Afficher l'image originale avec une limite de taille
  const maxCanvasWidth = 300; // Limite de la largeur du canvas
  const outputCanvas = document.getElementById("outputCanvas");
  const ctx = outputCanvas.getContext("2d");

  let scaledWidth = img.width;
  let scaledHeight = img.height;

  // Redimensionner si nécessaire pour respecter la largeur maximale
  if (img.width > maxCanvasWidth) {
    const scaleFactor = maxCanvasWidth / img.width;
    scaledWidth = maxCanvasWidth;
    scaledHeight = img.height * scaleFactor;
  }

  outputCanvas.width = scaledWidth;
  outputCanvas.height = scaledHeight;
  ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);

  // Transformer l'image (en arrière-plan pour le modèle)
  const transformedCanvas = transformImage(img);

  // Faire la prédiction
  const prediction = await predict(transformedCanvas);
  document.getElementById("prediction").textContent = `Type de vêtement : ${prediction}`;
});

// Fonction pour charger une image
function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}
