let predictionsData = [];
let currentPage = 0;
const perPage = 8;
const yahoo = "https://es.finance.yahoo.com/quote/";
const newspaper = "https://e01-expansion.uecdn.es/rss/portada.xml";

const tickerMap = {
        "ACS.MC": "ACS",
        "ACX.MC": "Acerinox",
        "AENA.MC": "Aena",
        "AMS.MC": "Amadeus",
        "ANA.MC": "Acciona",
        "ANE.MC": "Acciona Energía",
        "BBVA.MC": "BBVA",
        "BKT.MC": "Bankinter",
        "CABK.MC": "CaixaBank",
        "CLNX.MC": "Cellnex",
        "COL.MC": "Colonial",
        "ELE.MC": "Endesa",
        "ENG.MC": "Enagás",
        "FDR.MC": "Fluidra",
        "FER.MC": "Ferrovial",
        "GRF.MC": "Grifols",
        "IAG.MC": "IAG",
        "IBE.MC": "Iberdrola",
        "ITX.MC": "Inditex",
        "LOG.MC": "Logista",
        "MAP.MC": "Mapfre",
        "MRL.MC": "Merlin",
        "MTS.MC": "ArcelorMittal",
        "NTGY.MC": "Naturgy",
        "PUIG.MC": "Puig",
        "RED.MC": "Redeia",
        "SAB.MC": "Sabadell",
        "SAN.MC": "Santander",
        "TEF.MC": "Telefónica",
        "UNI.MC": "Unicaja"
}

function getFooter() {
    const date = new Date();
    const year = date.getFullYear();
    document.getElementById("footer").textContent = `Alex De La Haya © ${year} | 10**6`;
}


function getTitle() {
    const date = new Date();
    const dayOfWeek = date.getDay(); 

    let predictionDate = new Date(date);

    if(dayOfWeek == 6) {
        predictionDate.setDate(date.getDate() + 2); 

    }
    else if (dayOfWeek == 0) {
        predictionDate.setDate(date.getDate() + 1);
    }

    const day = predictionDate.getDate();
    const month = predictionDate.getMonth() + 1;

    document.getElementById("header").textContent = `Predicciones ${day}/${month}`;
}

async function getLatestNews() {
    try {
        const response = await fetch(newspaper);
        //missing logic ...
    }
    catch (error) {
        console.error("Error loading news:", error);
    }
}

async function getPredictions() {
    try {
        const response = await fetch("../data/predictions.json");
        predictionsData = await response.json();

        renderPredictions();
    } catch (error) {
        console.error("Error loading predictions:", error);
    }
}

function renderPredictions(direction = 0){

    const container = document.getElementById("predictions");

    if(direction !== 0){
        container.style.transform =
            direction > 0 ? "translateX(50px)" : "translateX(-50px)";
    }

    setTimeout(() => {

        container.innerHTML = "";

        const start = currentPage * perPage;
        const sortedData = [...predictionsData].sort((a, b) => b.proba - a.proba);
        const pageData = sortedData.slice(start, start + perPage);

        pageData.forEach(pred => {
            const url = yahoo + pred.ticker + "/"
            const cardLink = document.createElement("a")
            cardLink.href = url
            cardLink.target = "_blank"
            cardLink.rel = "noopener noreferrer";

            const card = document.createElement("div");
            card.className = "pred";

            const top = document.createElement("div");
            top.className = "pred-top";

            const name = document.createElement("h2");
            name.textContent = tickerMap[pred.ticker] || pred.ticker.replace(".MC", "");

            top.appendChild(name);

            const bottom = document.createElement("div");

            let action = (pred.pred === 1 ? "buy" : "sell");
            bottom.className = "pred-bottom " + action;

            action = action[0].toUpperCase() + action.slice(1);

            const prob = document.createElement("p");
            prob.textContent = `P(${action}|Xₜ) = ${pred.proba.toFixed(2)}`;

            bottom.appendChild(prob);

            card.appendChild(top);
            card.appendChild(bottom);

            cardLink.appendChild(card)
            container.appendChild(cardLink);

        });

        container.style.transform = "translateX(0)";

    }, 120);
}

document.getElementById("next").onclick = () => {

    const maxPage = Math.ceil(predictionsData.length / perPage) - 1;

    if(currentPage < maxPage){
        currentPage++;
        renderPredictions(1);
    }

};

document.getElementById("prev").onclick = () => {

    if(currentPage > 0){
        currentPage--;
        renderPredictions(-1);
    }

};

document.addEventListener("DOMContentLoaded", () => {

    getTitle();
    getPredictions();
    getLatestNews();
    getFooter();

});

