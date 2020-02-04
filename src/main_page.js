// Progress controller

var myDate = new Date();
var day = myDate.getDay();
console.log(day);

// bar secondary w-40
// secondary blue
// success  green
// warning  yellow
// danger  red

var progress = document.getElementById("progress_display");
// progress.setAttribute("class","bar success w-25");

var day_display = document.getElementById("day_display");

if(day==0){
    day_display.innerText = "Today is Sunday! No more time for this week!";
    progress.setAttribute("class","bar danger w-10");
    progress.innerText= "No more than 1 day";
}else if(day==1){
    day_display.innerText = "Today is Monday! Time to start coding!";
    progress.setAttribute("class","bar success w-100");
    progress.innerText = "6 days ready to go";
}else if(day==2){
    day_display.innerText = "Today is Tuesday! Things are getting hard, huh?";
    progress.setAttribute("class","bar success w-80");
    progress.innerText = "5 days are waiting for you"
}else if(day==3){
    day_display.innerText = "Today is Wednesday! Ready to cook some new code?";
    progress.setAttribute("class","bar warning w-60");
    progress.innerText = "4 days !!!";
}else if(day==4){
    day_display.innerText = "Today is Thursday! Hold on! Anything can be handled by coding!";
    progress.setAttribute("class","bar warning w-40");
    progress.innerText = "3 days are remaining";
}else if(day==5){
    day_display.innerText = "Today is Friday! Maybe code magister wants some coding break?";
    progress.setAttribute("class","bar warning w-30");
    progress.innerText = "2 days, they are so cute"
}else if(day==6){
    day_display.innerText = "Today is Saturday! Leetcode! rest and pace!";
    progress.setAttribute("class","bar danger w-20");
    progress.innerText = "1 day, 1 day, 1day";
}