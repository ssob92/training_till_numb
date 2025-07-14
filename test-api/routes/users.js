const router = require('express').Router();
const User = require('../models/user');

// CREATE
router.post('/', async (req, res) => {

    const user = new User(req.body);
    await user.save();
    res.status(201).json(user);
});

// READ (All)
router.get('/', async (req, res) => {

    const user = await User.find();
    res.json(user);
});

// UPDATE
router.put('/:id', async (req, res) => {

    const user = await User.findByIdAndUpdate(
        req.params.id,
        req.body,
        { new: true }
    );
    if (!user) return res.sendStatus(404);
    res.json(user);
});

// DELETE
router.delete('/:id', async (req, res) => {

    const user = await User.findByIdAndDelete(req.params.id);
    if (!user) return res.sendStatus(404);
    res.json({ message: 'User deleted' });
});

module.exports = router;