async function loadStatus(){

    const response=await fetch("/status");

    const data=await response.json();

    document.getElementById("room").innerHTML=

        data.room.estimated_room || "Unknown";

    document.getElementById("alert").innerHTML=

        data.alert || "Safe";

    let html="";

    data.objects.forEach(obj=>{

        html+=`

        <tr>

            <td>${obj.name}</td>

            <td>${obj.position}</td>

            <td>${obj.distance_cm ?? "-"}</td>

        </tr>

        `;

    });

    document.getElementById("objects").innerHTML=html;

}

setInterval(loadStatus,500);

loadStatus();

async function askAssistant(){

    const q=document.getElementById("question").value;

    const response=await fetch("/ask",{

        method:"POST",

        headers:{

            "Content-Type":"application/json"

        },

        body:JSON.stringify({

            question:q

        })

    });

    const data=await response.json();

    document.getElementById("answer").innerHTML=data.answer;

}