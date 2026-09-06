#include "passdialog.h"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QTabWidget>
#include <QVBoxLayout>

static QDoubleSpinBox* makeSpin(double lo, double hi, double step, double value, QWidget* parent) {
    auto* s = new QDoubleSpinBox(parent);
    s->setRange(lo, hi);
    s->setSingleStep(step);
    s->setValue(value);
    return s;
}

PassDialog::PassDialog(ShmHeader* h, QWidget* parent) : QDialog(parent), hdr(h) {
    setWindowTitle("Per-pass settings");
    resize(420, 360);

    auto* root = new QVBoxLayout(this);
    root->addWidget(new QLabel("Override global settings for individual passes.", this));
    tabs = new QTabWidget(this);
    root->addWidget(tabs);

    for (uint32_t i = 0; i < kMaxPasses; ++i) {
        auto* page = new QWidget(tabs);
        auto* form = new QFormLayout(page);
        PassTuning ps = hdr ? ShmResolvePass(hdr, i) : PassTuning{};

        rows[i].enabled = new QCheckBox("Override global settings", page);
        rows[i].enabled->setChecked(hdr && hdr->pass[i].overrideMask.load() != 0);
        form->addRow(rows[i].enabled);

        rows[i].intensity = makeSpin(0.0, 4.0, 0.05, ps.intensity, page);
        rows[i].localTone = makeSpin(0.0, 4.0, 0.05, ps.localTone, page);
        rows[i].localStructure = makeSpin(0.0, 4.0, 0.05, ps.localStructure, page);
        rows[i].skinStructure = makeSpin(-1.0, 4.0, 0.05, ps.skinStructure, page);
        rows[i].sharpness = makeSpin(0.0, 1.0, 0.05, ps.sharpness, page);

        form->addRow("Intensity", rows[i].intensity);
        form->addRow("Local tone", rows[i].localTone);
        form->addRow("Local structure", rows[i].localStructure);
        form->addRow("Skin structure", rows[i].skinStructure);
        form->addRow("Sharpness", rows[i].sharpness);

        const bool on = rows[i].enabled->isChecked();
        rows[i].intensity->setEnabled(on);
        rows[i].localTone->setEnabled(on);
        rows[i].localStructure->setEnabled(on);
        rows[i].skinStructure->setEnabled(on);
        rows[i].sharpness->setEnabled(on);

        connect(rows[i].enabled, &QCheckBox::toggled, this, [this, i](bool checked) {
            rows[i].intensity->setEnabled(checked);
            rows[i].localTone->setEnabled(checked);
            rows[i].localStructure->setEnabled(checked);
            rows[i].skinStructure->setEnabled(checked);
            rows[i].sharpness->setEnabled(checked);
            writePass(int(i));
        });
        connect(rows[i].intensity, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, i] { writePass(int(i)); });
        connect(rows[i].localTone, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, i] { writePass(int(i)); });
        connect(rows[i].localStructure, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, i] { writePass(int(i)); });
        connect(rows[i].skinStructure, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, i] { writePass(int(i)); });
        connect(rows[i].sharpness, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, i] { writePass(int(i)); });

        tabs->addTab(page, QString("Pass %1").arg(i + 1));
    }

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    root->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::close);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
}

void PassDialog::writePass(int pass) {
    if (!hdr || pass < 0 || pass >= int(kMaxPasses)) return;
    auto& r = rows[uint32_t(pass)];
    // The dialog still edits every field together, so it overrides them as a set. The protocol
    // carries a per-field mask so a later interface can be sparser without changing the contract.
    const uint32_t mask = r.enabled->isChecked()
                              ? (kOverrideIntensity | kOverrideLocalTone | kOverrideLocalStructure |
                                 kOverrideSkinStructure | kOverrideSharpness)
                              : 0u;
    hdr->pass[uint32_t(pass)].overrideMask.store(mask);
    hdr->pass[uint32_t(pass)].intensityBits.store(FloatToBits(float(r.intensity->value())));
    hdr->pass[uint32_t(pass)].localToneBits.store(FloatToBits(float(r.localTone->value())));
    hdr->pass[uint32_t(pass)].localStructureBits.store(FloatToBits(float(r.localStructure->value())));
    hdr->pass[uint32_t(pass)].skinStructureBits.store(FloatToBits(float(r.skinStructure->value())));
    hdr->pass[uint32_t(pass)].sharpnessBits.store(FloatToBits(float(r.sharpness->value())));
    hdr->controlSeq.fetch_add(1);
}