// JavaScript
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

  // Clear any existing content in the grid container.
  gridContainer.innerHTML = "";

  // Create grid items.
  for (let i = 0; i < gridSize * gridSize; i++) {
    const gridItem = document.createElement("div"); // Create a 'div' element.
    gridItem.classList.add("gridItem"); // Use the same class name as in your CSS.
    gridItem.style.width = `${squareSize}px`;
    gridItem.style.height = `${squareSize}px`;
    gridContainer.appendChild(gridItem);
  }
}

// Get grid size and create grid
const gridSize = askGridSize();
createGrid(gridSize);
//let div = document.querySelector(".gridItem");
//addEventListener("mouseenter");
