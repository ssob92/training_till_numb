
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const app = express();
app.use(express.json(), cors);

mongoose    
    .connect("mongodb://127.0.0.1:27017/userDB")
    .then(() => console.log('MongoDB connected'))
    .catch(console.error);

app.use('/users', require('./routes/users'));

const PORT = 3000;
app.listen(PORT, () => console.log('Listening on port 3000'));