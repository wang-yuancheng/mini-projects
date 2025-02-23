function getComputerChoice() {
  let num = Math.floor(Math.random() * 3);
  if (num === 0) {
    return "Rock";
  } else if (num === 1) {
    return "Paper";
  } else {
    return "Scissors";
  }
}

function getHumanChoice() {
  let HumanChoice = prompt("Choose either Rock, Paper or Scissors.");
  if (HumanChoice.toLowerCase() === "rock") {
    console.log("You chose Rock");
  } else if (HumanChoice.toLowerCase() === "paper") {
    console.log("You chose Paper");
  } else if (HumanChoice.toLowerCase() === "scissors") {
    HumanChoice.toLowerCase() === "scissors";
    console.log("You chose Scissors");
  } else {
    alert("Choose a valid choice!");
    getHumanChoice();
  }
  return HumanChoice;
}
