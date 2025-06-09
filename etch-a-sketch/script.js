function askGridSize() {
  let gridSize = parseInt(prompt("Choose a Grid Size between 1-100"), 10);
  if (isNaN(gridSize) || gridSize > 100 || gridSize <= 0) {
    alert("Choose a valid number!");
    return askGridSize();
  }
  return gridSize;
}

function createGrid(gridSize) {
  const gridContainer = document.querySelector(".gridContainer");
  containerWidth = gridContainer.clientWidth;
  squareSize = containerWidth / gridSize;

  // Clear any existing content in the grid container
  gridContainer.innerHTML = "";

  // Create grid items
  for (let i = 0; i < gridSize * gridSize; i++) {
    const gridItem = document.createElement("div");
    gridItem.classList.add("gridItem");
    gridItem.style.width = `${squareSize}px`;
    gridItem.style.height = `${squareSize}px`;
    
    gridItem.addEventListener("mouseenter", () => {
      gridItem.style.backgroundColor = getRandomColor();
    })
    
    gridContainer.appendChild(gridItem);
  }
}

function getRandomColor() {
  let r = Math.floor(Math.random() * 256);
  let g = Math.floor(Math.random() * 256);
  let b = Math.floor(Math.random() * 256);
  return `rgb(${r}, ${g}, ${b})`;
}

// Get grid size and create grid
const gridSize = askGridSize();
createGrid(gridSize);
