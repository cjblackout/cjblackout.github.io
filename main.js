const firebaseConfig = {
  apiKey: "AIzaSyDhNVW7CLZoS100IJRiNt_YH5hWorjhwdQ",
  authDomain: "message-collector-82b77.firebaseapp.com",
  databaseURL: "https://message-collector-82b77-default-rtdb.europe-west1.firebasedatabase.app",
  projectId: "message-collector-82b77",
  storageBucket: "message-collector-82b77.appspot.com",
  messagingSenderId: "269861500409",
  appId: "1:269861500409:web:63289ae6c096d63a4cf928",
  measurementId: "G-S2S8PW6W99"
};

// Reference messages
var messagesRef = firebase.database().ref('message-collector-82b77');

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

// listen for form submit
document.getElementById('contactForm').addEventListener('submit', submitForm);

function submitForm(e){
    e.preventDefault();

    // get values
    var name = getInputVal('name');
    var email = getInputVal('email');
    var message = getInputVal('message');

    saveMessage(name, email, message);
}

// Function to get get form values
function getInputVal(id){
    return document.getElementById(id).value;
}

// function to save message to firebase
function saveMessage(name, email, message){
    var newMessageRef = messagesRef.push();
    newMessageRef.set({
        name: name,
        email: email,
        message: message
    });
}
