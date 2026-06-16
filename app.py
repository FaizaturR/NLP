import streamlit as st
import torch
import torch.nn as nn

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="Machine Translation UII",
    page_icon="🌍",
    layout="centered"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================
# LOAD VOCAB
# =====================================

# Isi dengan vocab milikmu

# Contoh:
# id_vocab = {...}
# en_vocab = {...}

# Kalau sudah punya file pickle, bisa load dari sana


# reverse vocab

id_idx2word = {v: k for k, v in id_vocab.items()}
en_idx2word = {v: k for k, v in en_vocab.items()}


# =====================================
# MODEL
# =====================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_dim,
        emb_dim,
        hidden_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            input_dim,
            emb_dim
        )

        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim
        )

    def forward(self, src):

        embedded = self.embedding(src)

        outputs, hidden = self.rnn(embedded)

        return hidden



class Decoder(nn.Module):

    def __init__(
        self,
        output_dim,
        emb_dim,
        hidden_dim
    ):

        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            output_dim,
            emb_dim
        )

        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            output_dim
        )


    def forward(
        self,
        input,
        hidden
    ):

        input = input.unsqueeze(0)

        embedded = self.embedding(input)

        output, hidden = self.rnn(
            embedded,
            hidden
        )

        prediction = self.fc(
            output.squeeze(0)
        )

        return prediction, hidden




class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder
    ):

        super().__init__()

        self.encoder = encoder

        self.decoder = decoder



INPUT_DIM = len(id_vocab)

OUTPUT_DIM = len(en_vocab)

EMB_DIM = 128

HIDDEN_DIM = 256


encoder = Encoder(

    INPUT_DIM,

    EMB_DIM,

    HIDDEN_DIM

)


decoder = Decoder(

    OUTPUT_DIM,

    EMB_DIM,

    HIDDEN_DIM

)


model = Seq2Seq(

    encoder,

    decoder

).to(device)



# =====================================
# LOAD MODEL
# =====================================

model.load_state_dict(

    torch.load(
        "seq2seq_ind_en.pt",
        map_location=device
    )

)

model.eval()



# =====================================
# TRANSLATE
# =====================================

def translate_sentence(sentence):

    model.eval()

    tokens = sentence.lower().split()

    tokens = ["<sos>"] + tokens + ["<eos>"]


    src_indexes = [

        id_vocab.get(

            token,

            id_vocab["<unk>"]

        )

        for token in tokens

    ]


    src_tensor = torch.LongTensor(

        src_indexes

    ).unsqueeze(1).to(device)


    with torch.no_grad():

        hidden = model.encoder(
            src_tensor
        )


    trg_indexes = [

        en_vocab["<sos>"]

    ]


    for i in range(30):

        trg_tensor = torch.LongTensor(

            [trg_indexes[-1]]

        ).to(device)


        with torch.no_grad():

            output, hidden = model.decoder(

                trg_tensor,

                hidden

            )


        pred_token = output.argmax(1).item()


        trg_indexes.append(
            pred_token
        )


        if pred_token == en_vocab["<eos>"]:

            break


    trg_tokens = [

        en_idx2word[i]

        for i in trg_indexes

    ]


    return trg_tokens[1:-1]



# =====================================
# UI STREAMLIT
# =====================================

st.title("🌍 Machine Translation")

st.subheader("Indonesia → English")

st.write("Tugas UAS NLP")

st.write("Universitas Islam Indonesia (UII)")

st.divider()


sentence = st.text_area(

    "Masukkan kalimat Bahasa Indonesia",

    placeholder="Contoh: saya suka membaca buku"

)


if st.button("🔄 Translate"):


    if sentence == "":

        st.warning(
            "Masukkan kalimat terlebih dahulu!"
        )

    else:


        translation = translate_sentence(
            sentence
        )


        result = " ".join(
            translation
        )


        st.success(
            "Hasil Terjemahan"
        )


        st.write(result)



st.divider()


st.metric(

    label="BLEU Score",

    value="0.0515"

)