const buttonValues = [
    "AC", "+/-", "%", "÷", 
    "7", "8", "9", "×",
    "4", "5", "6", "-",
    "1", "2", "3", "+",
    "0", ".", "="
];
const rightSymbols = ["÷", "×", "-", "+", "="];
const topSymbols = ["AC", "+/-", "%"];

const display = document.getElementById("display")
//Operations
let A = 0;
let operator = null;
let B = null;

function clearAll() {
    A = 0;
    operator = null;
    B = null;
}

//create buttons with symbols
for(let i = 0; i < buttonValues.length; i++){
    let value = buttonValues[i];
    let button = document.createElement("button");
    button.innerText = value;

    if (value == "0"){
        button.style.width = "200px";
        button.style.gridColumn = "span 2";
    }

    //styling button colors
    if (rightSymbols.includes(value)) {
        button.style.backgroundColor = "#FF9500";
    }  else if (topSymbols.includes(value)){
        button.style.backgroundColor = "#D4D4D2";
        button.style.color = "#1C1C1C";
    }

    //process button clicks
    button.addEventListener("click", function() {
        if (rightSymbols.includes(value)) {
            if (value == "=") {
                if (A != null) {
                    B = display.value;
                    let numA = Number(A);
                    let numB = Number(B);
                    
                    if (operator == "÷") {
                        display.value = numA/numB;
                    }
                    else if (operator == "×") {
                        display.value = numA*numB;
                    }
                    else if (operator == "-") {
                        display.value = numA-numB;
                    }
                    else if (operator == "+") {
                        display.value = numA+numB;
                    }
                    clearAll()
                }
                
            } else {
                operator = value;
                A = display.value;
                display.value = "";
            }

        }
        else if (topSymbols.includes(value)) {
            if (value == "AC") {
                clearAll();
                display.value = "";
            }
            else if (value == "+/-") {
                if (display.value != "" && display.value != "0") {
                    if (display.value[0] == "-") {
                        display.value = display.value.slice(1);
                    }
                    else {
                        display.value = "-" + display.value
                    }
                }

            }
            else if (value == "%") {
                display.value = Number(display.value)/100;
            }
        }
        else {
            if (value == ".") {
                if (display.value != "" && !display.value.includes(value)) {
                    display.value += value;
                }
            } 
            else if (display.value == "0") {
                display.value = value
            }
            else {
                display.value += value;
            }
        }
    })

    //add each button to button div
    document.getElementById("buttons").appendChild(button);
}  