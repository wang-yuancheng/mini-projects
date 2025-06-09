function getComputerChoice() {
  let num = Math.floor(Math.random() * 3);
  if (num === 0) {
    return "rock";
  } else if (num === 1) {
    return "paper";
  } else {
    return "scissors";
  }
}

function getHumanChoice() {
  let humanChoice = prompt("Choose either Rock, Paper or Scissors.");
  humanChoice = humanChoice.toLowerCase();
  if (humanChoice === "rock") {
    console.log("You chose Rock");
  } else if (humanChoice === "paper") {
    console.log("You chose Paper");
  } else if (humanChoice === "scissors") {
    humanChoice === "scissors";
    console.log("You chose Scissors");
  } else {
    alert("Choose a valid choice!");
    return getHumanChoice();
  }
  return humanChoice;
}

function playRound(computerChoice, humanChoice) {
  console.log("Computer chose " + computerChoice);
  if (
    (computerChoice === "rock" && humanChoice === "scissors") ||
    (computerChoice === "scissors" && humanChoice === "paper") ||
    (computerChoice === "paper" && humanChoice === "rock")
  ) {
    alert("Computer chose: " + computerChoice);
    console.log("YOU LOSE!");
    alert("YOU LOSE!");
    computerScore += 1;
  } else if (
    (computerChoice === "rock" && humanChoice === "paper") ||
    (computerChoice === "scissors" && humanChoice === "rock") ||
    (computerChoice === "paper" && humanChoice === "scissors")
  ) {
    alert("Computer chose: " + computerChoice);
    console.log("YOU WIN!");
    alert("YOU WIN!");
    humanScore += 1;
  } else {
    alert("Computer chose: " + computerChoice);
    console.log("DRAW!");
    alert("DRAW");
  }

  choiceReplay = prompt(
    "Your score : " +
      humanScore +
      "\nComputer score: " +
      computerScore +
      "\nDo you want to play again, Y for Yes and any key for No"
  );
  if (
    choiceReplay.toLowerCase() === "y" ||
    choiceReplay.toLowerCase() === "yes"
  ) {
    getComputerChoice();
    console.log(playRound(getComputerChoice(), getHumanChoice()));
  } else {
    console.log(
      "Game Over! Your score : " +
        humanScore +
        "\nComputer score: " +
        computerScore
    );
  }
}

let humanScore = 0;
let computerScore = 0;
playRound(getComputerChoice(), getHumanChoice());
