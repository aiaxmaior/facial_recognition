import { useState, useMemo } from 'react'
import {
  Paper,
  TextInput,
  Group,
  Select,
  Badge,
  Text,
  Tooltip,
  Progress,
  Stack,
  Box,
  ActionIcon,
  Modal,
  ScrollArea,
  Code,
  Tabs
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { DataTable, DataTableSortStatus } from 'mantine-datatable'
import {
  IconSearch,
  IconFilter,
  IconUser,
  IconUserOff,
  IconUsers,
  IconMoodSmile,
  IconEye,
  IconX,
  IconScissors,
  IconPlayerPlay,
  IconInfoCircle
} from '@tabler/icons-react'
import { VideoPreview } from './VideoPreview'
import { PersonsPanel } from './PersonsPanel'
import type { ClipData } from '../types'

const PAGE_SIZE = 25

const EMOTION_COLORS: Record<string, string> = {
  happy: 'green',
  sad: 'indigo',
  angry: 'red',
  fear: 'yellow',
  surprise: 'grape',
  disgust: 'teal',
  neutral: 'gray',
  unknown: 'dark'
}

interface ClipsTableProps {
  clips: ClipData[]
}

export function ClipsTable({ clips }: ClipsTableProps) {
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [videoFilter, setVideoFilter] = useState<string | null>(null)
  const [emotionFilter, setEmotionFilter] = useState<string | null>(null)
  const [personFilter, setPersonFilter] = useState<string | null>(null)
  const [sortStatus, setSortStatus] = useState<DataTableSortStatus<ClipData>>({
    columnAccessor: 'sceneName',
    direction: 'asc'
  })
  const [selectedClip, setSelectedClip] = useState<ClipData | null>(null)
  const [detailsOpened, { open: openDetails, close: closeDetails }] = useDisclosure(false)
  const [videoPreviewClip, setVideoPreviewClip] = useState<ClipData | null>(null)
  const [videoPreviewOpened, { open: openVideoPreview, close: closeVideoPreview }] = useDisclosure(false)

  // Get unique values for filters
  const videoOptions = useMemo(() => 
    [...new Set(clips.map(c => c.videoName))].sort().map(v => ({ value: v, label: v })),
    [clips]
  )
  
  const emotionOptions = useMemo(() =>
    [...new Set(clips.map(c => c.dominantEmotion))].sort().map(e => ({ value: e, label: e })),
    [clips]
  )

  // Filter and sort data
  const filteredData = useMemo(() => {
    let data = [...clips]
    
    // Search filter
    if (search) {
      const searchLower = search.toLowerCase()
      data = data.filter(clip =>
        clip.sceneName.toLowerCase().includes(searchLower) ||
        clip.videoName.toLowerCase().includes(searchLower) ||
        (clip.caption && clip.caption.toLowerCase().includes(searchLower))
      )
    }
    
    // Video filter
    if (videoFilter) {
      data = data.filter(clip => clip.videoName === videoFilter)
    }
    
    // Emotion filter
    if (emotionFilter) {
      data = data.filter(clip => clip.dominantEmotion === emotionFilter)
    }
    
    // Person filter
    if (personFilter === 'with') {
      data = data.filter(clip => clip.personPresent)
    } else if (personFilter === 'without') {
      data = data.filter(clip => !clip.personPresent)
    }
    
    // Sort
    const { columnAccessor, direction } = sortStatus
    data.sort((a, b) => {
      const aVal = a[columnAccessor as keyof ClipData]
      const bVal = b[columnAccessor as keyof ClipData]
      
      if (aVal === null || aVal === undefined) return direction === 'asc' ? 1 : -1
      if (bVal === null || bVal === undefined) return direction === 'asc' ? -1 : 1
      
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return direction === 'asc' ? aVal - bVal : bVal - aVal
      }
      
      const aStr = String(aVal).toLowerCase()
      const bStr = String(bVal).toLowerCase()
      return direction === 'asc' ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr)
    })
    
    return data
  }, [clips, search, videoFilter, emotionFilter, personFilter, sortStatus])

  // Paginate
  const paginatedData = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE
    return filteredData.slice(start, start + PAGE_SIZE)
  }, [filteredData, page])

  const handleRowClick = (clip: ClipData) => {
    setSelectedClip(clip)
    openDetails()
  }

  const handleVideoPreview = (clip: ClipData) => {
    setVideoPreviewClip(clip)
    openVideoPreview()
  }

  const clearFilters = () => {
    setSearch('')
    setVideoFilter(null)
    setEmotionFilter(null)
    setPersonFilter(null)
  }

  const hasActiveFilters = search || videoFilter || emotionFilter || personFilter

  return (
    <>
      <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
        {/* Filters */}
        <Group mb="md" gap="sm">
          <TextInput
            placeholder="Search clips..."
            leftSection={<IconSearch size={16} />}
            value={search}
            onChange={(e) => setSearch(e.currentTarget.value)}
            style={{ flex: 1, maxWidth: 300 }}
          />
          <Select
            placeholder="Filter by video"
            leftSection={<IconFilter size={16} />}
            data={videoOptions}
            value={videoFilter}
            onChange={setVideoFilter}
            clearable
            searchable
            style={{ width: 200 }}
          />
          <Select
            placeholder="Filter by emotion"
            leftSection={<IconMoodSmile size={16} />}
            data={emotionOptions}
            value={emotionFilter}
            onChange={setEmotionFilter}
            clearable
            style={{ width: 160 }}
          />
          <Select
            placeholder="Person filter"
            leftSection={<IconUser size={16} />}
            data={[
              { value: 'with', label: 'With person' },
              { value: 'without', label: 'Without person' }
            ]}
            value={personFilter}
            onChange={setPersonFilter}
            clearable
            style={{ width: 160 }}
          />
          {hasActiveFilters && (
            <ActionIcon variant="subtle" color="gray" onClick={clearFilters}>
              <IconX size={16} />
            </ActionIcon>
          )}
        </Group>

        <Text size="sm" c="dimmed" mb="sm">
          Showing {filteredData.length} of {clips.length} clips
        </Text>

        {/* Data Table */}
        <DataTable
          withTableBorder={false}
          borderRadius="md"
          striped
          highlightOnHover
          records={paginatedData}
          columns={[
            {
              accessor: 'sceneName',
              title: 'Scene',
              sortable: true,
              width: 180,
              render: (clip) => (
                <Text size="sm" fw={500} truncate>
                  {clip.sceneName}
                </Text>
              )
            },
            {
              accessor: 'videoName',
              title: 'Video',
              sortable: true,
              width: 150,
              render: (clip) => (
                <Badge variant="light" color="cerulean" size="sm">
                  {clip.videoName}
                </Badge>
              )
            },
            {
              accessor: 'personPresent',
              title: 'Person',
              sortable: true,
              width: 80,
              textAlign: 'center',
              render: (clip) => (
                clip.personPresent ? (
                  <Tooltip label={`${clip.maxPersons} person(s) detected`}>
                    <IconUser size={18} color="#40c057" />
                  </Tooltip>
                ) : (
                  <IconUserOff size={18} color="#868e96" />
                )
              )
            },
            {
              accessor: 'detectionCoverage',
              title: 'Coverage',
              sortable: true,
              width: 120,
              render: (clip) => (
                <Tooltip label={`${(clip.detectionCoverage * 100).toFixed(1)}% detection coverage`}>
                  <Progress
                    value={clip.detectionCoverage * 100}
                    color={clip.detectionCoverage > 0.6 ? 'green' : clip.detectionCoverage > 0.3 ? 'yellow' : 'gray'}
                    size="sm"
                  />
                </Tooltip>
              )
            },
            {
              accessor: 'dominantEmotion',
              title: 'Emotion',
              sortable: true,
              width: 110,
              render: (clip) => (
                <Badge
                  variant="light"
                  color={EMOTION_COLORS[clip.dominantEmotion] || 'gray'}
                  size="sm"
                >
                  {clip.dominantEmotion}
                </Badge>
              )
            },
            {
              accessor: 'meanValence',
              title: 'Valence',
              sortable: true,
              width: 100,
              render: (clip) => (
                <Text
                  size="sm"
                  c={clip.meanValence > 0 ? 'green' : clip.meanValence < -0.3 ? 'red' : 'dimmed'}
                  fw={500}
                >
                  {clip.meanValence.toFixed(2)}
                </Text>
              )
            },
            {
              accessor: 'meanArousal',
              title: 'Arousal',
              sortable: true,
              width: 100,
              render: (clip) => (
                <Text
                  size="sm"
                  c={clip.meanArousal > 0.6 ? 'orange' : 'dimmed'}
                  fw={500}
                >
                  {clip.meanArousal.toFixed(2)}
                </Text>
              )
            },
            {
              accessor: 'durationSeconds',
              title: 'Duration',
              sortable: true,
              width: 90,
              render: (clip) => (
                <Text size="sm" c="dimmed">
                  {clip.durationSeconds.toFixed(1)}s
                </Text>
              )
            },
            {
              accessor: 'caption',
              title: 'Caption',
              width: 80,
              textAlign: 'center',
              render: (clip) => (
                clip.caption ? (
                  <Badge variant="dot" color="green" size="sm">Yes</Badge>
                ) : clip.captionError ? (
                  <Tooltip label={clip.captionError}>
                    <Badge variant="dot" color="red" size="sm">Error</Badge>
                  </Tooltip>
                ) : (
                  <Badge variant="dot" color="gray" size="sm">None</Badge>
                )
              )
            },
            {
              accessor: 'actions',
              title: '',
              width: 80,
              render: (clip) => (
                <Group gap={4}>
                  <Tooltip label="Play video">
                    <ActionIcon
                      variant="subtle"
                      color="green"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleVideoPreview(clip)
                      }}
                    >
                      <IconPlayerPlay size={16} />
                    </ActionIcon>
                  </Tooltip>
                  <Tooltip label="View details">
                    <ActionIcon
                      variant="subtle"
                      color="cerulean"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleRowClick(clip)
                      }}
                    >
                      <IconEye size={16} />
                    </ActionIcon>
                  </Tooltip>
                </Group>
              )
            }
          ]}
          sortStatus={sortStatus}
          onSortStatusChange={setSortStatus}
          totalRecords={filteredData.length}
          recordsPerPage={PAGE_SIZE}
          page={page}
          onPageChange={setPage}
          onRowClick={({ record }) => handleRowClick(record)}
          rowStyle={() => ({ cursor: 'pointer' })}
          minHeight={400}
          noRecordsText="No clips match the current filters"
        />
      </Paper>

      {/* Detail Modal */}
      <Modal
        opened={detailsOpened}
        onClose={closeDetails}
        title={
          <Group gap="sm">
            <IconScissors size={20} color="#3395f3" />
            <Text fw={600}>{selectedClip?.sceneName}</Text>
          </Group>
        }
        size="xl"
        centered
      >
        {selectedClip && (
          <Tabs defaultValue="details" variant="outline">
            <Tabs.List mb="md">
              <Tabs.Tab value="details" leftSection={<IconInfoCircle size={16} />}>
                Details
              </Tabs.Tab>
              <Tabs.Tab 
                value="persons" 
                leftSection={<IconUsers size={16} />}
                disabled={!selectedClip.personPresent}
                rightSection={
                  selectedClip.personPresent && selectedClip.maxPersons > 0 ? (
                    <Badge size="xs" variant="filled" color="grape">
                      {selectedClip.maxPersons}
                    </Badge>
                  ) : null
                }
              >
                Persons
              </Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="details">
              <ScrollArea h={500}>
                <Stack gap="md">
                  {/* Basic Info */}
                  <Box>
                    <Text size="sm" fw={600} c="dimmed" mb="xs">Basic Information</Text>
                    <Group gap="lg">
                      <Stack gap={2}>
                        <Text size="xs" c="dimmed">Video</Text>
                        <Badge color="cerulean">{selectedClip.videoName}</Badge>
                      </Stack>
                      <Stack gap={2}>
                        <Text size="xs" c="dimmed">Duration</Text>
                        <Text fw={500}>{selectedClip.durationSeconds.toFixed(2)}s</Text>
                      </Stack>
                      <Stack gap={2}>
                        <Text size="xs" c="dimmed">Frames</Text>
                        <Text fw={500}>{selectedClip.totalFrames}</Text>
                      </Stack>
                      <Stack gap={2}>
                        <Text size="xs" c="dimmed">Person Present</Text>
                        <Badge color={selectedClip.personPresent ? 'green' : 'gray'}>
                          {selectedClip.personPresent ? `Yes (${selectedClip.maxPersons})` : 'No'}
                        </Badge>
                      </Stack>
                    </Group>
                  </Box>

                  {/* Detection Stats */}
                  <Box>
                    <Text size="sm" fw={600} c="dimmed" mb="xs">Detection Metrics</Text>
                    <Group gap="lg">
                      <Stack gap={2}>
                        <Text size="xs" c="dimmed">Coverage</Text>
                        <Progress
                          value={selectedClip.detectionCoverage * 100}
                          color="cerulean"
                          size="lg"
                          style={{ width: 100 }}
                        />
                        <Text size="xs">{(selectedClip.detectionCoverage * 100).toFixed(1)}%</Text>
                      </Stack>
                      <Stack gap={2}>
                        <Text size="xs" c="dimmed">Avg Confidence</Text>
                        <Text fw={500}>{(selectedClip.avgConfidence * 100).toFixed(1)}%</Text>
                      </Stack>
                    </Group>
                  </Box>

                  {/* Emotion Data */}
                  <Box>
                    <Text size="sm" fw={600} c="dimmed" mb="xs">Emotion Analysis</Text>
                    <Group gap="lg" mb="sm">
                      <Stack gap={2}>
                        <Text size="xs" c="dimmed">Dominant</Text>
                        <Badge color={EMOTION_COLORS[selectedClip.dominantEmotion] || 'gray'}>
                          {selectedClip.dominantEmotion}
                        </Badge>
                      </Stack>
                      <Stack gap={2}>
                        <Text size="xs" c="dimmed">Valence</Text>
                        <Text fw={500} c={selectedClip.meanValence > 0 ? 'green' : 'red'}>
                          {selectedClip.meanValence.toFixed(3)}
                        </Text>
                      </Stack>
                      <Stack gap={2}>
                        <Text size="xs" c="dimmed">Arousal</Text>
                        <Text fw={500}>{selectedClip.meanArousal.toFixed(3)}</Text>
                      </Stack>
                      <Stack gap={2}>
                        <Text size="xs" c="dimmed">Pain/Pleasure</Text>
                        <Text fw={500}>{selectedClip.painPleasureScore.toFixed(3)}</Text>
                      </Stack>
                    </Group>
                    
                    {/* Emotion Distribution */}
                    {selectedClip.emotionDistribution && Object.keys(selectedClip.emotionDistribution).length > 0 && (
                      <Box>
                        <Text size="xs" c="dimmed" mb="xs">Distribution</Text>
                        <Stack gap={4}>
                          {Object.entries(selectedClip.emotionDistribution)
                            .sort(([,a], [,b]) => b - a)
                            .map(([emotion, value]) => (
                              <Group key={emotion} gap="xs">
                                <Text size="xs" w={70}>{emotion}</Text>
                                <Progress
                                  value={value}
                                  color={EMOTION_COLORS[emotion] || 'gray'}
                                  size="sm"
                                  style={{ flex: 1 }}
                                />
                                <Text size="xs" w={40}>{value.toFixed(1)}%</Text>
                              </Group>
                            ))
                          }
                        </Stack>
                      </Box>
                    )}
                  </Box>

                  {/* Demographics */}
                  {(selectedClip.ageEstimates.length > 0 || selectedClip.genderEstimates.length > 0) && (
                    <Box>
                      <Text size="sm" fw={600} c="dimmed" mb="xs">Demographics</Text>
                      <Group gap="lg">
                        {selectedClip.ageEstimates.length > 0 && (
                          <Stack gap={2}>
                            <Text size="xs" c="dimmed">Age Estimates</Text>
                            <Group gap={4}>
                              {selectedClip.ageEstimates.map((age, i) => (
                                <Badge key={i} size="sm" variant="outline">{age}</Badge>
                              ))}
                            </Group>
                          </Stack>
                        )}
                        {selectedClip.genderEstimates.length > 0 && (
                          <Stack gap={2}>
                            <Text size="xs" c="dimmed">Gender Estimates</Text>
                            <Group gap={4}>
                              {selectedClip.genderEstimates.map((gender, i) => (
                                <Badge key={i} size="sm" variant="outline">{gender}</Badge>
                              ))}
                            </Group>
                          </Stack>
                        )}
                      </Group>
                    </Box>
                  )}

                  {/* Caption */}
                  <Box>
                    <Text size="sm" fw={600} c="dimmed" mb="xs">Caption</Text>
                    {selectedClip.caption ? (
                      <Paper p="sm" bg="dark.7" radius="sm">
                        <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
                          {selectedClip.caption}
                        </Text>
                      </Paper>
                    ) : selectedClip.captionError ? (
                      <Badge color="red">{selectedClip.captionError}</Badge>
                    ) : (
                      <Text size="sm" c="dimmed" fs="italic">No caption generated</Text>
                    )}
                  </Box>

                  {/* Paths */}
                  <Box>
                    <Text size="sm" fw={600} c="dimmed" mb="xs">File Paths</Text>
                    <Stack gap={4}>
                      <Code block>{selectedClip.scenePath}</Code>
                      {selectedClip.vlmPath && <Code block>{selectedClip.vlmPath}</Code>}
                    </Stack>
                  </Box>

                  {/* Video Preview Button in Modal */}
                  <Box>
                    <ActionIcon
                      variant="filled"
                      color="cerulean"
                      size="lg"
                      onClick={() => {
                        closeDetails()
                        handleVideoPreview(selectedClip)
                      }}
                    >
                      <IconPlayerPlay size={20} />
                    </ActionIcon>
                    <Text size="xs" c="dimmed" mt={4}>Play video</Text>
                  </Box>
                </Stack>
              </ScrollArea>
            </Tabs.Panel>

            <Tabs.Panel value="persons">
              <ScrollArea h={500}>
                <PersonsPanel sceneName={selectedClip.sceneName} />
              </ScrollArea>
            </Tabs.Panel>
          </Tabs>
        )}
      </Modal>

      {/* Video Preview Modal */}
      <VideoPreview
        opened={videoPreviewOpened}
        onClose={closeVideoPreview}
        videoUrl={videoPreviewClip ? `/api/video/scene/${videoPreviewClip.sceneName}` : ''}
        title={videoPreviewClip?.sceneName || ''}
        subtitle={videoPreviewClip ? `${videoPreviewClip.videoName} • ${videoPreviewClip.durationSeconds.toFixed(1)}s` : ''}
      />
    </>
  )
}
