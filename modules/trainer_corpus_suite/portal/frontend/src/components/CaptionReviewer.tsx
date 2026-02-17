import { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Card,
  Group,
  Stack,
  Title,
  Text,
  Image,
  Textarea,
  Button,
  Badge,
  Select,
  ActionIcon,
  Paper,
  Loader,
  Center,
  Progress,
  Tooltip,
  Modal,
  TagsInput,
  Divider,
  rem,
  Grid,
} from '@mantine/core'
import { useHotkeys } from '@mantine/hooks'
import { notifications } from '@mantine/notifications'
import {
  IconArrowLeft,
  IconArrowRight,
  IconDeviceFloppy,
  IconTrash,
  IconPhoto,
  IconTags,
  IconRefresh,
  IconCheck,
  IconX,
  IconKeyboard,
} from '@tabler/icons-react'

interface CaptionDataset {
  name: string
  path: string
  imageCount: number
  hasMetadata: boolean
}

interface CaptionImage {
  id: string
  filename: string
  path: string
  caption: string
  negativePrompt: string
  tags: string[]
  hasCaption: boolean
}

interface CaptionReviewerProps {
  // Optional initial dataset
  initialDataset?: string
}

export function CaptionReviewer({ initialDataset }: CaptionReviewerProps) {
  const [datasets, setDatasets] = useState<CaptionDataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState<string | null>(initialDataset || null)
  const [images, setImages] = useState<CaptionImage[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [deleteModalOpen, setDeleteModalOpen] = useState(false)
  
  // Editable fields
  const [editedCaption, setEditedCaption] = useState('')
  const [editedNegativePrompt, setEditedNegativePrompt] = useState('')
  const [editedTags, setEditedTags] = useState<string[]>([])
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  
  // Stats
  const [totalImages, setTotalImages] = useState(0)
  const [captionedCount, setCaptionedCount] = useState(0)

  // Fetch available datasets
  useEffect(() => {
    fetchDatasets()
  }, [])

  // Fetch images when dataset changes
  useEffect(() => {
    if (selectedDataset) {
      fetchImages(selectedDataset)
    }
  }, [selectedDataset])

  // Update edited fields when image changes
  useEffect(() => {
    const img = images[currentIndex]
    if (img) {
      setEditedCaption(img.caption)
      setEditedNegativePrompt(img.negativePrompt)
      setEditedTags(img.tags)
      setHasUnsavedChanges(false)
    }
  }, [currentIndex, images])

  // Track changes
  useEffect(() => {
    const img = images[currentIndex]
    if (img) {
      const changed = editedCaption !== img.caption || 
                      editedNegativePrompt !== img.negativePrompt
      setHasUnsavedChanges(changed)
    }
  }, [editedCaption, editedNegativePrompt, currentIndex, images])

  const fetchDatasets = async () => {
    try {
      const res = await fetch('/api/caption-datasets')
      const data = await res.json()
      setDatasets(data.datasets)
      
      // Auto-select first dataset if available and none selected
      if (data.datasets.length > 0 && !selectedDataset) {
        setSelectedDataset(data.datasets[0].name)
      }
    } catch (err) {
      notifications.show({
        title: 'Error',
        message: 'Failed to load datasets',
        color: 'red',
      })
    }
  }

  const fetchImages = async (datasetName: string) => {
    setLoading(true)
    try {
      // Load all images (we'll paginate later if needed)
      const res = await fetch(`/api/caption-dataset/${datasetName}?limit=1000`)
      const data = await res.json()
      setImages(data.images)
      setTotalImages(data.total)
      setCaptionedCount(data.images.filter((img: CaptionImage) => img.hasCaption).length)
      setCurrentIndex(0)
    } catch (err) {
      notifications.show({
        title: 'Error',
        message: 'Failed to load images',
        color: 'red',
      })
    } finally {
      setLoading(false)
    }
  }

  const saveCaption = async () => {
    const img = images[currentIndex]
    if (!img || !selectedDataset) return

    setSaving(true)
    try {
      const res = await fetch(`/api/caption-dataset/${selectedDataset}/${img.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          caption: editedCaption,
          negative_prompt: editedNegativePrompt,
        }),
      })

      if (!res.ok) throw new Error('Failed to save')

      // Update local state
      const updatedImages = [...images]
      updatedImages[currentIndex] = {
        ...img,
        caption: editedCaption,
        negativePrompt: editedNegativePrompt,
        tags: editedCaption.split(',').map(t => t.trim()).filter(Boolean).slice(0, 20),
        hasCaption: true,
      }
      setImages(updatedImages)
      setCaptionedCount(updatedImages.filter(i => i.hasCaption).length)
      setHasUnsavedChanges(false)

      notifications.show({
        title: 'Saved',
        message: 'Caption updated successfully',
        color: 'green',
        icon: <IconCheck size={16} />,
      })
    } catch (err) {
      notifications.show({
        title: 'Error',
        message: 'Failed to save caption',
        color: 'red',
      })
    } finally {
      setSaving(false)
    }
  }

  const deleteImage = async () => {
    const img = images[currentIndex]
    if (!img || !selectedDataset) return

    try {
      const res = await fetch(`/api/caption-dataset/${selectedDataset}/${img.id}`, {
        method: 'DELETE',
      })

      if (!res.ok) throw new Error('Failed to delete')

      // Remove from local state
      const updatedImages = images.filter((_, i) => i !== currentIndex)
      setImages(updatedImages)
      setTotalImages(updatedImages.length)
      setCaptionedCount(updatedImages.filter(i => i.hasCaption).length)
      
      // Adjust index if needed
      if (currentIndex >= updatedImages.length) {
        setCurrentIndex(Math.max(0, updatedImages.length - 1))
      }

      setDeleteModalOpen(false)
      notifications.show({
        title: 'Deleted',
        message: 'Image removed from dataset',
        color: 'orange',
      })
    } catch (err) {
      notifications.show({
        title: 'Error',
        message: 'Failed to delete image',
        color: 'red',
      })
    }
  }

  const goToNext = useCallback(() => {
    if (currentIndex < images.length - 1) {
      setCurrentIndex(currentIndex + 1)
    }
  }, [currentIndex, images.length])

  const goToPrevious = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1)
    }
  }, [currentIndex])

  const saveAndNext = useCallback(async () => {
    await saveCaption()
    goToNext()
  }, [saveCaption, goToNext])

  // Keyboard shortcuts
  useHotkeys([
    ['ArrowLeft', goToPrevious],
    ['ArrowRight', goToNext],
    ['ctrl+s', (e) => { e.preventDefault(); saveCaption() }],
    ['ctrl+Enter', (e) => { e.preventDefault(); saveAndNext() }],
  ])

  const currentImage = images[currentIndex]

  return (
    <Box>
      {/* Header */}
      <Group justify="space-between" mb="md">
        <Group gap="md">
          <Select
            placeholder="Select dataset"
            data={datasets.map(d => ({ value: d.name, label: `${d.name} (${d.imageCount})` }))}
            value={selectedDataset}
            onChange={setSelectedDataset}
            w={250}
          />
          <ActionIcon variant="subtle" onClick={() => selectedDataset && fetchImages(selectedDataset)}>
            <IconRefresh size={18} />
          </ActionIcon>
        </Group>

        <Group gap="xs">
          <Tooltip label="Keyboard: ← → navigate, Ctrl+S save, Ctrl+Enter save & next">
            <Badge leftSection={<IconKeyboard size={14} />} variant="light" color="gray">
              Shortcuts
            </Badge>
          </Tooltip>
          {totalImages > 0 && (
            <Badge variant="light" color="cerulean">
              {captionedCount}/{totalImages} captioned
            </Badge>
          )}
        </Group>
      </Group>

      {/* Progress bar */}
      {totalImages > 0 && (
        <Progress
          value={(captionedCount / totalImages) * 100}
          color="cerulean"
          size="sm"
          mb="md"
          radius="xl"
        />
      )}

      {loading ? (
        <Center h={400}>
          <Loader size="lg" color="cerulean" />
        </Center>
      ) : !selectedDataset || images.length === 0 ? (
        <Center h={400}>
          <Stack align="center" gap="md">
            <IconPhoto size={64} color="#666" />
            <Text c="dimmed">
              {!selectedDataset ? 'Select a dataset to start reviewing' : 'No images in this dataset'}
            </Text>
          </Stack>
        </Center>
      ) : (
        <Grid gutter="lg">
          {/* Image Panel */}
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Card withBorder p="md" h="100%">
              <Stack gap="md">
                {/* Navigation */}
                <Group justify="space-between">
                  <ActionIcon
                    variant="light"
                    size="lg"
                    onClick={goToPrevious}
                    disabled={currentIndex === 0}
                  >
                    <IconArrowLeft size={20} />
                  </ActionIcon>
                  
                  <Text fw={500}>
                    {currentIndex + 1} / {images.length}
                  </Text>
                  
                  <ActionIcon
                    variant="light"
                    size="lg"
                    onClick={goToNext}
                    disabled={currentIndex === images.length - 1}
                  >
                    <IconArrowRight size={20} />
                  </ActionIcon>
                </Group>

                {/* Image */}
                <Box
                  style={{
                    borderRadius: rem(8),
                    overflow: 'hidden',
                    background: '#000',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: rem(400),
                  }}
                >
                  <Image
                    src={`/api/caption-image/${selectedDataset}/${currentImage.id}`}
                    alt={currentImage.filename}
                    fit="contain"
                    h={400}
                  />
                </Box>

                {/* Image info */}
                <Group justify="space-between">
                  <Text size="sm" c="dimmed" truncate style={{ maxWidth: '70%' }}>
                    {currentImage.filename}
                  </Text>
                  <Badge
                    color={currentImage.hasCaption ? 'green' : 'gray'}
                    variant="light"
                    size="sm"
                  >
                    {currentImage.hasCaption ? 'Has caption' : 'No caption'}
                  </Badge>
                </Group>
              </Stack>
            </Card>
          </Grid.Col>

          {/* Caption Panel */}
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Card withBorder p="md" h="100%">
              <Stack gap="md" h="100%">
                <Group justify="space-between">
                  <Title order={4}>Caption</Title>
                  {hasUnsavedChanges && (
                    <Badge color="yellow" variant="light">Unsaved changes</Badge>
                  )}
                </Group>

                <Textarea
                  placeholder="Enter caption/prompt..."
                  value={editedCaption}
                  onChange={(e) => setEditedCaption(e.currentTarget.value)}
                  minRows={8}
                  maxRows={12}
                  autosize
                  styles={{
                    input: {
                      fontFamily: 'JetBrains Mono, monospace',
                      fontSize: rem(13),
                    },
                  }}
                />

                <Divider label="Negative Prompt" labelPosition="left" />

                <Textarea
                  placeholder="Enter negative prompt (optional)..."
                  value={editedNegativePrompt}
                  onChange={(e) => setEditedNegativePrompt(e.currentTarget.value)}
                  minRows={3}
                  maxRows={6}
                  autosize
                  styles={{
                    input: {
                      fontFamily: 'JetBrains Mono, monospace',
                      fontSize: rem(13),
                    },
                  }}
                />

                <Divider label="Tags Preview" labelPosition="left" />

                <Group gap="xs" style={{ flexWrap: 'wrap' }}>
                  {editedCaption.split(',').slice(0, 15).map((tag, i) => (
                    <Badge key={i} variant="outline" size="sm" color="gray">
                      {tag.trim().substring(0, 30)}
                    </Badge>
                  ))}
                  {editedCaption.split(',').length > 15 && (
                    <Badge variant="outline" size="sm" color="dimmed">
                      +{editedCaption.split(',').length - 15} more
                    </Badge>
                  )}
                </Group>

                {/* Actions */}
                <Group mt="auto" justify="space-between">
                  <Button
                    color="red"
                    variant="subtle"
                    leftSection={<IconTrash size={16} />}
                    onClick={() => setDeleteModalOpen(true)}
                  >
                    Delete
                  </Button>

                  <Group gap="sm">
                    <Button
                      variant="light"
                      leftSection={<IconDeviceFloppy size={16} />}
                      onClick={saveCaption}
                      loading={saving}
                      disabled={!hasUnsavedChanges}
                    >
                      Save
                    </Button>
                    <Button
                      leftSection={<IconCheck size={16} />}
                      onClick={saveAndNext}
                      loading={saving}
                    >
                      Save & Next
                    </Button>
                  </Group>
                </Group>
              </Stack>
            </Card>
          </Grid.Col>
        </Grid>
      )}

      {/* Delete Confirmation Modal */}
      <Modal
        opened={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        title="Delete Image"
        centered
      >
        <Stack gap="md">
          <Text>
            Are you sure you want to delete this image and its caption? This action cannot be undone.
          </Text>
          <Text size="sm" c="dimmed">
            {currentImage?.filename}
          </Text>
          <Group justify="flex-end" gap="sm">
            <Button variant="subtle" onClick={() => setDeleteModalOpen(false)}>
              Cancel
            </Button>
            <Button color="red" onClick={deleteImage}>
              Delete
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Box>
  )
}
